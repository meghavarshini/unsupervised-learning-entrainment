# use this script to generate dyads from the MultiCAT data
# for the purposes of this script, we assume the following:
# 1. every trial contains a transcript with timestamps of each utterance
# as well as the speaker and the addressee
# 2. every trial has associated wav files for each speaker that
# can be identified using the name of the transcript file + speaker info
# NOTE: if speaker is described by role, there should be an associated speaker ID
#   e.g. E000689 = 'transporter'
# NOTE: RUN THIS FILE FROM THE BASE DIRECTORY

import pandas as pd
from pathlib import Path
import re
import subprocess as sp
import numpy as np
import sys

############ edit this line if you have trouble with home path #######
sys.path.append("/Users/meghavarshinikrishnaswamy/github/unsupervised-learning-entrainment")
from feats.feat_extract_nopre import final_feat_calculate_multicat
import copy



def loop_through_data(combined_files_dict, save_dir):
    # go to an individual file in the data
    for fpath, combined_file in combined_files_dict.items():
        print(combined_file)
        # speakers and addressees in CSV MUST be named their unique ID
        all_speakers = combined_file.speaker.unique().tolist()

        combined_file = combined_file.sort_values(by=["start"], ascending=True)

        speakers = []
        # get the list of speakers from this file
        for speaker in all_speakers:
            if len(all_speakers) > 1:
                all_speakers.remove(speaker)
                for other in all_speakers:
                    speakers.append([speaker, other])

        # iterate through pairs of speakers
        for speaker_pair in speakers:
            print(f"SPEAKER PAIR IS: {speaker_pair}")
            # holder for all acoustic features
            all_features_for_this_list = []
            # holder for concatenated feature lists
            features_for_normalizing = []

            # create savename for this file
            fname = fpath.stem
            savename = f"{save_dir}/{fname}_{speaker_pair[0]}-{speaker_pair[1]}.csv"

            # row num
            row_num = 0
            # holder for all utt rows in overall feats file
            all_row_ends = []

            # go through the combined file
            for i, row in combined_file.iterrows():

                if i < len(combined_file) - 1:
                    next_row = combined_file.iloc[i+1]
                    if row.speaker != next_row.speaker:
                        # print(f"row {i}: {row.speaker}; row {i + 1}: {row.speaker}")
                        extract_me = id_whether_to_extract(row, next_row, speaker_pair)
                        if extract_me:
                            # print(f"extract_me is True!")
                            # extract the relevant components of this row and the following row

                            # extract for this row
                            this_row_feats, this_row_copy = run_opensmile_over_utterance(row, fpath)
                            # extract for following row
                            next_row_feats, next_row_copy = run_opensmile_over_utterance(next_row, fpath)

                            # add to row num for this utt
                            this_utt_ends = [row_num, row_num + len(this_row_copy)]
                            row_num += len(this_row_copy)

                            # add to row num for next utt
                            next_utt_ends = [row_num, row_num + len(next_row_copy)]
                            row_num += len(next_row_copy)

                            # add to counter for all rows
                            all_row_ends.append(this_utt_ends)
                            all_row_ends.append(next_utt_ends)

                            # add feature copies to features_for_normalizing
                            features_for_normalizing.extend(this_row_copy)
                            features_for_normalizing.extend(next_row_copy)

                            # save these features to holders for later change
                            all_features_for_this_list.append(this_row_feats)
                            all_features_for_this_list.append(next_row_feats)

            features_for_normalizing = np.asarray(features_for_normalizing)

            if np.shape(features_for_normalizing)[0] > 0:
                # run f0 and intensity normalization over the features for normalizing
                all_raw_norm_feat, all_norm_feat_dim =  calc_opensmile_feats(features_for_normalizing)

                # all_features
                all_features = None

                for i in range(len(all_features_for_this_list)):
                    if i % 2 == 0:
                        this_utt_feats = np.asarray(all_features_for_this_list[i])
                        next_utt_feats = np.asarray(all_features_for_this_list[i+1])

                        whole_func_feat1 = final_feat_calculate_multicat(
                            all_row_ends[i], all_raw_norm_feat, all_norm_feat_dim
                        )
                        whole_func_feat2 = final_feat_calculate_multicat(
                            all_row_ends[i+1], all_raw_norm_feat, all_norm_feat_dim
                        )
                        whole_func_feat = np.hstack((whole_func_feat1, whole_func_feat2))
                        if all_features is None:
                            all_features = whole_func_feat
                        else:
                            all_features = np.vstack((all_features, whole_func_feat))

                # save the features
                with open(savename, 'w') as savefile:
                    for item in all_features:
                        savefile.write(",".join([str(part) for part in item]))
                        savefile.write("\n")


def run_opensmile_over_utterance(row, base_file):
    # get start and end times, name of files
    start = row.start
    end = row.end
    speaker = row.speaker

    # get the name of the audio
    # combined files are of format: Trial-T000604_Team-TM000202_combined.txt
    filepath = base_file.parent
    fname = base_file.stem
    # re.sub(pattern, repl, string, count=0, flags=0)¶
    if "NA" in fname:
        audio_in_name = re.sub(r"NA", speaker, fname) + ".wav"
    else:
        audio_in_name = fname + ".wav"
    audio_out_name = f"{audio_in_name}_{start}-{end}.wav"
    audio_out = filepath / audio_out_name
    audio_out = str(audio_out)

    length = end - start

    # extract this short file to run feature extraction on
    sp.run(["ffmpeg", "-ss", str(start), "-i", f"{str(filepath)}/{audio_in_name}",
            "-t", str(length), "-c", "copy", audio_out])

    feats_out = filepath / f"{speaker}_{start}-{end}.csv"
    feats_out = str(feats_out)

    # run opensmile over a particular utterance
    # feats to extract
    OPENSMILE_CONFIG_BASELINE = "/Users/meghavarshinikrishnaswamy/github/unsupervised-learning-entrainment/feats/emobase2010_haoqi_revised.conf"

    # $(OUTPUT_DIR)/%_features_raw_baseline.csv: $(OUTPUT_DIR)/%.wav
    # 	SMILExtract -C $(OPENSMILE_CONFIG_BASELINE) -I $< -O $@
    # extract the features with opensmile

    # todo Megh: change the SMILExtract location
    sp.run(["/Users/meghavarshinikrishnaswamy/github/unsupervised-learning-entrainment/opensmile-3.0/bin/SMILExtract", "-C", OPENSMILE_CONFIG_BASELINE,
           "-I", audio_out, "-O", feats_out])

    # read in acoustic features
    acoustic_feats = []
    with open(feats_out, 'r') as feats:
        feats.readline()
        for line in feats:
            line = line.strip()
            line = line.split(",")
            line = [np.float64(item) for item in line]
            acoustic_feats.append(line)

    # make a copy of this
    acoustic_feats_2 = copy.deepcopy(acoustic_feats)

    # remove generated files
    sp.run(["rm", feats_out])
    sp.run(["rm", audio_out])

    return acoustic_feats, acoustic_feats_2


def calc_opensmile_feats(feat_data):
    """
    ## feature selection and normalization (original comments)

    - remove the mean for mfcc
    - normalize for pitch = log(f_0/u_0)
    - normalize for loudness
    """
    # f0 normalization
    f0 = np.copy(feat_data[:, 70])

    # replace 0 in f0 with nan
    f0[f0 == 0.0] = np.nan
    f0_mean = np.nanmean(f0)

    #if norm:
    f0[~np.isnan(f0)] = np.log2(f0[~np.isnan(f0)] / f0_mean)

    f0 = np.reshape(f0, (-1, 1))
    # f0_de normalization
    f0_de = np.copy(feat_data[:, 74])
    f0_de[f0_de == 0.0] = np.nan

    #if norm:
    f0_de_mean = np.nanmean(np.absolute(f0_de))
    f0_de[~np.isnan(f0_de)] = np.log2(
        np.absolute(f0_de[~np.isnan(f0_de)] / f0_de_mean)
    )

    f0_de = np.reshape(f0_de, (-1, 1))

    # intensity normalization
    intensity = np.copy(feat_data[:, 2])
    #if norm:
    int_mean = np.mean(intensity)
    intensity = intensity / int_mean

    intensity = np.reshape(intensity, (-1, 1))
    # intensity_de normalization
    intensity_de = np.copy(feat_data[:, 36])

    #if norm:
    int_de_mean = np.mean(intensity_de)
    intensity_de = intensity_de / int_de_mean

    intensity_de = np.reshape(intensity_de, (-1, 1))
    # feat_idx = range(3,34) + range(37, 68)   with spectral de
    feat_idx = list(range(3, 34))
    mfcc_etc = np.copy(feat_data[:, feat_idx])
    #if norm:
    mfcc_etc_mean = np.mean(mfcc_etc, axis=0)
    mfcc_etc_mean.reshape(-1, 1)
    mfcc_etc_norm = mfcc_etc - mfcc_etc_mean
    # else:
    #     mfcc_etc_norm = np.copy(mfcc_etc)

    # jitter and shimmer normalization
    idx_jitter_shimmer = [71, 72, 73]
    jitter_shimmer = np.copy(feat_data[:, idx_jitter_shimmer])
    jitter_shimmer[jitter_shimmer == 0.0] = np.nan

    #if norm:
    jitter_shimmer_mean = np.nanmean(jitter_shimmer, axis=0)
    jitter_shimmer_mean.reshape(-1, 1)
    jitter_shimmer_norm = jitter_shimmer - jitter_shimmer_mean
    # else:
    #     jitter_shimmer_norm = jitter_shimmer

    ##-----------------------------------------------------------------------
    ## function calculation
    ##-----------------------------------------------------------------------
    all_raw_norm_feat = np.hstack(
        (
            f0,
            f0_de,
            intensity,
            intensity_de,
            jitter_shimmer_norm,
            mfcc_etc_norm,
        )
    )

    # feature dimension
    all_raw_feat_dim = all_raw_norm_feat.shape[1]

    return all_raw_norm_feat, all_raw_feat_dim


def id_whether_to_extract(row, following_row, speaker_pair):
    if row.speaker in speaker_pair:
        speaker = row.speaker
        spk_idx = speaker_pair.index(row.speaker)
        # find which speaker is the speaker of first row in pair
        # and which we expect to be the second
        if spk_idx == 1:
            spk2_idx = 0
        else:
            spk2_idx = 1
        listener = speaker_pair[spk2_idx]

        # check if conditions are met
        if row.addressee == listener and following_row.speaker == listener and following_row.addressee == speaker:
            return True

    return False


if __name__ == "__main__":
    # todo: also change the path to SMILExtract in line 163
    # get location to savedir
    savedir = "/Users/meghavarshinikrishnaswamy/transcripts/files_for_pipeline/output"

    # get location to dir with files of interest
    data_dir = Path("/Users/meghavarshinikrishnaswamy/transcripts/files_for_pipeline")  # todo: change to path with files

    all_files_of_interest = {}

    for datafile in data_dir.iterdir():
        if datafile.suffix == ".csv":
            # read in the file as a pd df
            this_file = pd.read_csv(datafile, sep="\t")  # \t for tab delimited text files
            all_files_of_interest[datafile] = this_file

    # go through all files and generate output
    loop_through_data(all_files_of_interest, savedir)