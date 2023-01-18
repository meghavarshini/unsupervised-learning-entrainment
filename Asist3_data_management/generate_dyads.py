# use this script to generate dyads from the MultiCAT data
# for the purposes of this script, we assume the following:
# 1. every trial contains a transcript with timestamps of each utterance
# as well as the speaker and the addressee
# 2. every trial has associated wav files for each speaker that
# can be identified using the name of the transcript file + speaker info
# NOTE: if speaker is described by role, there should be an associated speaker ID
#   e.g. E000689 = 'transporter'

import numpy as np
import pandas as pd
import asist2_transcript_manage
from pathlib import Path


def loop_through_data(combined_files_list, save_dir):
    # go to an individual file in the data
    for combined_file in combined_files_list:
        # get the list of speakers from this file
        speakers = []  # todo: addme

        # iterate through pairs of speakers
        for speaker_pair in speakers:
            # holder for all acoustic features
            all_features_for_this_list = []

            # create savename for this file
            savename = f"{save_dir}/{combined_file}_{speaker_pair[0]}-{speaker_pair[1]}.csv"

            # go through the combined file
            for i, row in combined_file.iterrows():

                if i < len(combined_file) - 1:
                    extract_me = id_whether_to_extract(row, combined_file.iloc[i+1], speaker_pair)
                    if extract_me:
                        # extract the relevant components of this row and the following row
                        # extract for this row
                        this_row_feats = pass  # todo: addme
                        # extract for following row
                        next_row_feats = pass  # todo: addme

                        saved_feats = this_row_feats + next_row_feats
                        all_features_for_this_list.append(saved_feats)

            # save the features
            with open(savename, 'w') as savefile:
                for row_pair in all_features_for_this_list:
                    savefile.write(row_pair)


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
    savedir = "."

    # get location to dir with files of interest
    data_dir = Path(".")  # change to path with files

    all_files_of_interest = []

    for datafile in data_dir.iterdir():
        # read in the file as a pd df
        this_file = pd.read_csv(datafile, sep="\t")  # \t for tab delimited text files
        all_files_of_interest.append(this_file)

    # go through all files and generate output
    loop_through_data(all_files_of_interest, savedir)