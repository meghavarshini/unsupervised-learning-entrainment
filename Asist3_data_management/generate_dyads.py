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


def loop_through_data(combined_files_dict, save_dir):
    # go to an individual file in the data
    for fpath, combined_file in combined_files_dict.items():

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

            # create savename for this file
            fname = fpath.stem
            savename = f"{save_dir}/{fname}_{speaker_pair[0]}-{speaker_pair[1]}.csv"

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
                            this_row_feats = run_opensmile_over_utterance(row, fpath)
                            # extract for following row
                            next_row_feats = run_opensmile_over_utterance(next_row, fpath)

                            saved_feats = this_row_feats + next_row_feats
                            all_features_for_this_list.append(saved_feats)

            # save the features
            with open(savename, 'w') as savefile:
                for row_pair in all_features_for_this_list:
                    savefile.write(row_pair)


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

    # extract this short file to run feature extraction on
    sp.run(["ffmpeg", "-ss", str(start), "-i", f"{str(filepath)}/{audio_in_name}",
            "-to", str(end), "-c", "copy", audio_out])

    feats_out = filepath / f"{speaker}_{start}-{end}"
    feats_out = str(feats_out)

    # run opensmile over a particular utterance
    # feats to extract
    OPENSMILE_CONFIG_BASELINE = "feats/emobase2010_haoqi_revised.conf"

    # $(OUTPUT_DIR)/%_features_raw_baseline.csv: $(OUTPUT_DIR)/%.wav
    # 	SMILExtract -C $(OPENSMILE_CONFIG_BASELINE) -I $< -O $@
    # extract the features with opensmile

    sp.run(["/home/jculnan/opensmile-3.0/bin/SMILExtract", "-C", OPENSMILE_CONFIG_BASELINE,
           "-I", audio_out, "-O", feats_out])

    # read in acoustic features
    acoustic_feats = []
    with open(feats_out, 'r') as feats:
        c = 0
        for line in feats:
            c += 1
            acoustic_feats.append(line)
        if c > 1:
            exit("Multiple lines in this file: this is not expected")

    # remove generated files
    sp.run(["rm", feats_out])
    sp.run(["rm", audio_out])

    return acoustic_feats


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
    # get location to savedir
    savedir = "test_output"

    # get location to dir with files of interest
    data_dir = Path("test_data")  # todo: change to path with files

    all_files_of_interest = {}

    for datafile in data_dir.iterdir():
        if datafile.suffix == ".csv":
            # read in the file as a pd df
            this_file = pd.read_csv(datafile, sep=",")  # \t for tab delimited text files
            all_files_of_interest[datafile] = this_file

    # go through all files and generate output
    loop_through_data(all_files_of_interest, savedir)