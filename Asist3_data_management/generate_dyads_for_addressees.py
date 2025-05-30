'''
Use this script to generate dyads from the MultiCAT data.
It subsets dyads if the addressee is not 'all', and splits them up
by speaker and addressee.
NOTE: This file must be run from the Asist3_data_management folder
for the purposes of this script, we assume the following:
1. For every trial, there exists one transcript file with the following 
columns: "start	end	speaker	addressee	transcript", and 3 associated wav files. 
The file name contains the IDs for trial, speaker, and team.
2. The values column "speaker" should match the uniqueIDs of the participants,
not roles e.g. E000689, not 'transporter'
4. OpenSMILE and ffmpeg are installed, and the path to `SMILExtract`
has been added to $PATH (see documentation:
https://audeering.github.io/opensmile/get-started.html)
'''

import pandas as pd
from pathlib import Path
import re
import subprocess as sp
import numpy as np
import sys
import os
import copy
import argparse

def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup")
	parser.add_argument("--input_directory",
						default="./multicat_addressee_files_for_dyad_generation",
						help="directory where the input files are stored")
	return parser

############ Fix for issues with paths #######
## Get the absolute path of the parent directory
current_directory = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
## Add the parent directory to the system path
sys.path.append(parent_dir)

#set path for opensmile config file
OPENSMILE_CONFIG_BASELINE = parent_dir + "/scripts_and_config/emobase2010_haoqi_revised.conf"


## feature extraction function
from fisher_model_training.feat_extract_nopre import final_feat_calculate_multicat

def loop_through_data(combined_transcript_dict, save_dir):
	'''
	This function loops over every file added to a dictionary of unique files
	and saves the feature set to the user-defined location
	'''
	## go to an individual file in the data
	for fpath, combined_transcript in combined_transcript_dict.items():
		print("filepath in loop: ", fpath)
		## print("file contents: ", combined_file)
		## speakers and addressees in CSV MUST be named their unique ID
		all_speakers = combined_transcript.speaker.unique().tolist()
		print("Speakers: ", all_speakers)
		
		## arrange all turns in order
		combined_transcript = combined_transcript.sort_values(by=["start"], ascending=True)

		## get paired list of speakers from all_speakers, to create dyads
		speaker_pairs = []
		for speaker in all_speakers:
			if len(all_speakers) > 1:
				all_speakers.remove(speaker)
				for other in all_speakers:
					speaker_pairs.append([speaker, other])

		## iterate through pairs of speakers
		for speaker_pair in speaker_pairs:
			print(f"SPEAKER PAIR IS: {speaker_pair}")
			## holder for all acoustic features
			all_features_for_this_list = []
			## holder for concatenated feature lists
			features_for_normalizing = []

			## create savename for this file
			fname = fpath.stem
			savename = f"{save_dir}/{fname}_{speaker_pair[0]}-{speaker_pair[1]}.csv"

			## row num
			row_num = 0
			## holder for all utt rows in overall feats file
			all_row_ends = []

			## go through the combined file
			for i, row in combined_transcript.iterrows():

				if i < len(combined_transcript) - 1:
					next_row = combined_transcript.iloc[i+1]
					#ToDo: add else condition
					if row.speaker != next_row.speaker:
						## print(f"row {i}: {row.speaker}; row {i + 1}: {next_row.speaker}")
						extract_me = id_whether_to_extract(row, next_row, speaker_pair)
						if extract_me:
							print(f"extract_me for row {i} is True!")
							## extract the relevant components of this row and the following row

							## extract for this row
							this_row_feats, this_row_copy = run_opensmile_over_utterance(row, fpath)
							## extract for following row
							next_row_feats, next_row_copy = run_opensmile_over_utterance(next_row, fpath)

							## add to row num for this utt
							this_utt_ends = [row_num, row_num + len(this_row_copy)]
							row_num += len(this_row_copy)

							## add to row num for next utt
							next_utt_ends = [row_num, row_num + len(next_row_copy)]
							row_num += len(next_row_copy)

							## add to counter for all rows
							all_row_ends.append(this_utt_ends)
							all_row_ends.append(next_utt_ends)

							## add feature copies to features_for_normalizing
							features_for_normalizing.extend(this_row_copy)
							features_for_normalizing.extend(next_row_copy)

							## save these features to holders for later change
							all_features_for_this_list.append(this_row_feats)
							all_features_for_this_list.append(next_row_feats)

			features_for_normalizing = np.asarray(features_for_normalizing)

			if np.shape(features_for_normalizing)[0] <= 0:
				print("empty feature holder")
			elif np.shape(features_for_normalizing)[0] > 0:
				## run f0 and intensity normalization over the features for normalizing
				all_raw_norm_feat, all_norm_feat_dim =  calc_opensmile_feats(features_for_normalizing)

				## all_features
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

				## save the features
				print("savename:", savename)
				with open(savename, 'w') as savefile:
					for item in all_features:
						savefile.write(",".join([str(part) for part in item]))
						savefile.write("\n")


def run_opensmile_over_utterance(row, base_file):
	## get start and end times, name of files
	start = row.start
	end = row.end
	speaker = row.speaker

	## get the name of the audio
	## combined files are of format: Trial-T000604_Team-TM000202_combined.txt
	filepath = base_file.parent
	fname = base_file.stem

	## re.sub(pattern, repl, string, count=0, flags=0)¶
	if "NA" in fname:
		audio_in_name = re.sub(r"NA", speaker, fname) + ".wav"
	else:
		audio_in_name = fname + ".wav"
	audio_out_name = f"{audio_in_name}_{start}-{end}.wav"
	audio_out = filepath / audio_out_name
	audio_out = str(audio_out)

	length = end - start

	## extract this short file to run feature extraction on
	sp.run(["ffmpeg", "-ss", str(start), "-i", f"{str(filepath)}/{audio_in_name}",
			"-t", str(length), "-c", "copy","-y", audio_out])

	feats_out = filepath / f"{speaker}_{start}-{end}.csv"
	feats_out = str(feats_out)

	## run opensmile over a particular utterance

	## extract the features with opensmile
	sp.run(["SMILExtract", "-C", OPENSMILE_CONFIG_BASELINE,
			"-I", audio_out, "-O", feats_out])

	## read in acoustic features
	acoustic_feats = []
	with open(feats_out, 'r') as feats:
		feats.readline()
		for line in feats:
			line = line.strip()
			line = line.split(",")
			line = [np.float64(item) for item in line]
			acoustic_feats.append(line)

	## make a copy of this
	acoustic_feats_2 = copy.deepcopy(acoustic_feats)

	## remove generated files
	sp.run(["rm", feats_out])
	sp.run(["rm", audio_out])

	return acoustic_feats, acoustic_feats_2


def calc_opensmile_feats(feat_data):
	"""
	### feature selection and normalization (original comments)

	- remove the mean for mfcc
	- normalize for pitch = log(f_0/u_0)
	- normalize for loudness
	"""
	## f0 normalization
	f0 = np.copy(feat_data[:, 70])

	## replace 0 in f0 with nan
	f0[f0 == 0.0] = np.nan
	f0_mean = np.nanmean(f0)

	#if norm:
	f0[~np.isnan(f0)] = np.log2(f0[~np.isnan(f0)] / f0_mean)

	f0 = np.reshape(f0, (-1, 1))
	## f0_de normalization
	f0_de = np.copy(feat_data[:, 74])
	f0_de[f0_de == 0.0] = np.nan

	#if norm:
	f0_de_mean = np.nanmean(np.absolute(f0_de))
	f0_de[~np.isnan(f0_de)] = np.log2(
		np.absolute(f0_de[~np.isnan(f0_de)] / f0_de_mean)
	)

	f0_de = np.reshape(f0_de, (-1, 1))

	## intensity normalization
	intensity = np.copy(feat_data[:, 2])
	#if norm:
	int_mean = np.mean(intensity)
	intensity = intensity / int_mean

	intensity = np.reshape(intensity, (-1, 1))
	## intensity_de normalization
	intensity_de = np.copy(feat_data[:, 36])

	#if norm:
	int_de_mean = np.mean(intensity_de)
	intensity_de = intensity_de / int_de_mean

	intensity_de = np.reshape(intensity_de, (-1, 1))
	## feat_idx = range(3,34) + range(37, 68)   with spectral de
	feat_idx = list(range(3, 34))
	mfcc_etc = np.copy(feat_data[:, feat_idx])
	#if norm:
	mfcc_etc_mean = np.mean(mfcc_etc, axis=0)
	mfcc_etc_mean.reshape(-1, 1)
	mfcc_etc_norm = mfcc_etc - mfcc_etc_mean
	## else:
	#	 mfcc_etc_norm = np.copy(mfcc_etc)

	## jitter and shimmer normalization
	idx_jitter_shimmer = [71, 72, 73]
	jitter_shimmer = np.copy(feat_data[:, idx_jitter_shimmer])
	jitter_shimmer[jitter_shimmer == 0.0] = np.nan

	## if norm:
	jitter_shimmer_mean = np.nanmean(jitter_shimmer, axis=0)
	jitter_shimmer_mean.reshape(-1, 1)
	jitter_shimmer_norm = jitter_shimmer - jitter_shimmer_mean
	## else:
	#	 jitter_shimmer_norm = jitter_shimmer

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
	## feature dimension
	all_raw_feat_dim = all_raw_norm_feat.shape[1]

	return all_raw_norm_feat, all_raw_feat_dim


def id_whether_to_extract(row, following_row, speaker_pair):
	if row.speaker in speaker_pair:
		speaker = row.speaker
		spk_idx = speaker_pair.index(row.speaker)
		## find which speaker is the speaker of first row in pair
		## and which we expect to be the second
		if spk_idx == 1:
			spk2_idx = 0
		else:
			spk2_idx = 1
		listener = speaker_pair[spk2_idx]
		print(f"speaker: {speaker}, intended listener: {listener}, addressee: {row.addressee}")

		## check if conditions are met: a given utterance and the successive turn 
		## have the same interlocutors: 
		if row.addressee == listener and following_row.speaker == listener and following_row.addressee == speaker:
			return True

	return False


if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()

	## get location to dir with files of interest

	input_dir = Path(args.input_directory).resolve()
	
	## Create the full path for the "output" folder
	output_dir = Path.cwd().resolve() / "multicat_addressee_feats/test"

	print(f"input: {input_dir}\n output: {output_dir}")
	#Check if output directory exists, if not, create it:   
	if not output_dir.exists():
		output_dir.mkdir(parents=True, exist_ok=True)
		print(f"Could not find specified output directory {output_dir}. Creating directory...")
	else:
		print(f"Specified output directory {output_dir} already exists. Continuing...")
	
	#Create a dictionary of file names and acoustic features dataframes
	all_files_of_interest = {}

	for datafile in input_dir.iterdir():
		if datafile.suffix == ".csv":
			## read in the file as a pd df
			this_file = pd.read_csv(datafile, sep=",")  # \t for tab delimited text files
			all_files_of_interest[datafile] = this_file
	
	## go through all files and generate output
	loop_through_data(all_files_of_interest, output_dir)