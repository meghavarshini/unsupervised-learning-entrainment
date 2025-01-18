# ------------------------------------------------------------------------
# Name : feat_extract_nopre.py
# Author : Md Nasir
# Date   : 03-11-19
# Description : resample to 16k Hz, and run openSMILE to extract features
# ------------------------------------------------------------------------
from __future__ import division
import sys, os
import csv, pickle
from os.path import basename
import pandas as pd
import numpy as np
import argparse
import subprocess
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pdb
from normutils import normalizefeats
from normutils import final_feat_calculate
from normutils import func_calculate
from normutils import get_neighbor

# -----------------
def_wav = '/home/nasir/data/Fisher/fe_03_03892.csv'
config_path = 'emobase2010_revised.conf'
ivec_segment_out_dir = '/home/nasir/workspace/acoustic/triplet/switchboard/ivec_segments/'


# feat_extract_triplet_ivec.py  --audio_file  /home/nasir/data/Fisher/fe_03_03892.csv

IPU_gap=500
writing=True   # set True for getting functionals
extract=False 
seg = True # set True if want to create segments for ivector extraction
triplet = False  # set True if getting triplets for training, 
# False means we just want pairs to evaluate

# For posidon -----------------------------------
# transcript_dir='/home/nasir/data/Fisher/transcripts/'
# raw_dir = '/home/nasir/data/Fisher/raw_feats/'

#Assess where this pkl file is created:
if triplet:
	ivec_norm_dict_nospk = pickle.load( open("vectors/ivector_normalized_nospk_py3.pkl","rb"))


# ------------------------------------------------------------------------
# Params Setup				 
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--audio_file', type=str, required=False, default=def_wav,
					help='File path of the input audio file')
parser.add_argument('--openSMILE_config', type=str, required=False, default=config_path,
					help='config file of openSMILE')
parser.add_argument('--raw_path', type=str, required=True,
					help='raw feature folder path')
parser.add_argument('--trandir_path', type=str, required=True,
					help='transcript folder path')
parser.add_argument('--output_path', type=str, required=True,
					help='output folder path')
parser.add_argument('--norm', type=str, required=False, default=True, 
					help='do session level normalization or not')
parser.add_argument('--window_size', required=False, type=float, default=10)

args = parser.parse_args()

INPUT_audio      = args.audio_file
CONFIG_openSMILE = args.openSMILE_config
raw_dir          = args.raw_path
transcript_dir   = args.trandir_path
OUTPUT_path      = args.output_path

window_size      = args.window_size
norm             = args.norm

print('Current audio file:  '+ INPUT_audio)

#----------------------------------------------------------------
#---------------------------------------------------------------------
# check if file is wav or not
#---------------------------------------------------------------------
if extract:
	not_wav = False
	if basename(INPUT_audio).split('.')[-1] != 'wav':
		not_wav = True
		print('convert to .wav file...' )
		# cmd2wav = 'sox ' + INPUT_audio +' '+ basename(INPUT_audio).split('.')[-2]+'.wav'
		cmd2wav = 'sph2pipe -f rif ' + INPUT_audio +' '+ basename(INPUT_audio).split('.')[-2]+'.wav'
		subprocess.call(cmd2wav, shell  = True)
		INPUT_audio = basename(INPUT_audio).split('.')[-2]+'.wav'
		file_to_be_removed = basename(INPUT_audio).split('.')[-2]+'.wav'
	# ------------------------------------------------------------------------
	# downsample audio to 16kHz and convert to mono (unless file already downsampled)
	# ------------------------------------------------------------------------
	cmd_check_sample_rate = 'sox --i -r '+ INPUT_audio
	sample_rate = subprocess.getstatusoutput(cmd_check_sample_rate)
	not_16k = False
	if sample_rate[1] != '16000':
		not_16k = True
		print("Resampling to 16k ... ")
		output_16k_audio = 'resampled--' + os.path.basename(INPUT_audio)
		cmd_resample = 'sox %s -b 16 -c 1 -r 16k %s dither -s' %(INPUT_audio, output_16k_audio)
		subprocess.call(cmd_resample, shell  = True)
		# replace variable with downsampled audio
		#INPUT_audio = ''.join(output_16k_audio.split('--')[1:])
		INPUT_audio = output_16k_audio

	# # ------------------------------------------------------------------------
	# # extract feature use openSMILE
	# # ------------------------------------------------------------------------
	if not os.path.exists(OUTPUT_path):
		os.makedirs(OUTPUT_path)
	if not_16k:
		csv_file_name = raw_dir+'/'+basename(INPUT_audio).split('.wav')[0].split('--')[1] + '.csv'
	else:
		csv_file_name = raw_dir+'/'+basename(INPUT_audio).split('.wav')[0] + '.csv'
	print("Using openSMILE to extract features ... ")
	cmd_feat = 'SMILExtract -nologfile -C %s -I %s -O %s' %(CONFIG_openSMILE, INPUT_audio, csv_file_name)
	subprocess.call(cmd_feat, shell  = True)

	# delete resampled audio file
	if not_wav:
		os.remove(file_to_be_removed)
	if not_16k:
		os.remove(output_16k_audio)

# ------------------------------------------------------------------------
# load transcript timings
# ------------------------------------------------------------------------
## TODO: load the file from trans/, store it in an array in start, end, spk (A or B) fmt

spk_list=[]
ext='.sph'



if extract:
	sess_id = basename(INPUT_audio).split(ext)[0].split('--')[1] 
else:
	sess_id = basename(INPUT_audio).split(ext)[0]

transcript = transcript_dir + '/' + sess_id + '.txt'
trans = open(transcript).readlines()
last_spk = None
for line in trans:
	start, stop, spk = line.split(':')[0].split(' ')

	# if the turn duration is less than 0.5s, then ignore it. Excludes very short turns
	# Why are they doing this tho? One-word utterances are an issue?
	if float(stop)-float(start) < IPU_gap/1000:
		continue
	spk_list.append([start, stop, spk])
	if not last_spk:
		last_spk = spk
ini = last_spk
# ------------------------------------------------------------------------
# functional calculation: this has to be  PER Utterance (for entrainment)
# ------------------------------------------------------------------------
# frame length and overlap size in seconds # frame_len = window_size/0.01 # frame_shift_len = shift_size/0.01
if extract:
	csv_file_name = raw_dir + '/' + basename(INPUT_audio).split(ext)[0].split('--')[1]  + '.csv'
else:
	csv_file_name = raw_dir + '/' + basename(INPUT_audio).split(ext)[0]  + '.csv'

# read csv feature file
csv_feat = pd.read_csv(csv_file_name, dtype=np.float32)
csv_feat = csv_feat.values.copy()

featdata = np.copy(csv_feat)
# convert the first column index to int index
sample_index = list(map(int,list((featdata[:,0]))))

# def turn_level_index(spk_list, sample_index):
	# '''generate indices for different turns'''
turn_level_index_list=[]
#find out if this means- 2 speakers are the same
s2_found=True
gap_found=True
# pdb.set_trace()
# iterate over list of turns
for spch in spk_list:
	# convert to miliseconds
	start = int(float(spch[0])/0.01)
	stop = int(float(spch[1])/0.01)
	spk = spch[2]
	if not turn_level_index_list:
		# add acoustic features between 2 start and end time points in utterance
		turn_level_index_list = [sample_index[start:stop]]
		last_stop =stop
		continue
	if spk==last_spk:
		# if same speaker, add features to previous entry, if gap is small
		if start-last_stop < IPU_gap/10:
			turn_level_index_list[-1].extend(sample_index[start:stop])

		else:
			# if  IPU gap is sufficient:
			if s2_found:
				if gap_found:
					turn_level_index_list[-1]=sample_index[start:stop]
				else:
					turn_level_index_list.append(sample_index[start:stop])
					
			else:
				turn_level_index_list.append(sample_index[start:stop])
				s2_found=True		
			gap_found=True		
	else:
		# if IPU gap is insufficient:
		if not gap_found:
			turn_level_index_list.append(turn_level_index_list[-1])
		gap_found=False
		s2_found=False
		turn_level_index_list.append(sample_index[start:stop])
	last_stop=stop
	last_spk = spk

#remove last item, so that every utterance is paired with another
if len(turn_level_index_list)%2==1:
	turn_level_index_list=turn_level_index_list[:-1]

s1_list=[]
s2_list=[]
for i, itm in enumerate(turn_level_index_list):
	# if i is even, i.e for every 2 utterances:
	if i%2==0:
		s1_list.append(itm)
	else:
		s2_list.append(itm)

if seg:
	segf = open(ivec_segment_out_dir + sess_id +'.seg', 'w')
	for i, turn2 in enumerate(s2_list):
		start = turn2[0]
		spk = format(i, '04d')
		stop = turn2[-1]
		# Check if this is an ivector extraction thing:
        #utt_id = sess_id +  '_' + str(0.01*float(start)) + '-' + str(0.01*float(stop))
		utt_id_seg = sess_id + '-' +  spk +  '_' + format(int(float(start)), '06d') + '-' + format(int(float(stop)), '06d') 
		# print(utt_id_seg + ' ' + sess_id + '-' + spk+  ' ' + format(0.01*float(start), '0.2f') + ' ' + format(0.01*float(stop), '0.2f') + '\n')
		# pdb.set_trace()
		segf.write(utt_id_seg + ' ' + sess_id + '-' + spk+  ' '
				   + format(0.01*float(start), '0.2f') + ' ' + format(0.01*float(stop), '0.2f') + '\n')
	segf.close()


##-----------------------------------------------------------------------
## function calculation
##-----------------------------------------------------------------------
all_raw_norm_feat = normalizefeats(featdata, norm)
# feature dimension
all_raw_feat_dim = all_raw_norm_feat.shape[1]

whole_func_feat1 = final_feat_calculate(s1_list, all_raw_norm_feat, all_raw_feat_dim)
whole_func_feat2 = final_feat_calculate(s2_list, all_raw_norm_feat, all_raw_feat_dim)

# 'fe_03_04835_151590-153150'
# run a for loop for s2_list, 

if triplet:
	ivec_segment_out_dir = '/home/nasir/data/Fisher/ivec_segments/'
	count = 0
	found =[]
	whole_func_feat3 = np.array([])

	# seg=True
	# if seg:
	# 	segf = open(ivec_segment_out_dir + sess_id +'.seg', 'w')

	for i, turn2 in enumerate(s2_list):
		start = turn2[0]
		stop = turn2[-1]+1
		utt_id = sess_id +  '_' + str(int(10*float(start))) + '-' + str(int(10*float(stop))) 
		# utt_id_part = sess_id +  '_' + str(int(10*float(start))) + '-'
		# if seg:
		# 	utt_id_seg = str(i)+ '-' + sess_id +  '_' + str(int(10*float(start))) + '-' + str(int(10*float(stop))) 
		# 	segf.write(utt_id_seg + ' ' + sess_id + ' ' + str(int(10*float(start))) + ' '+ str(int(10*float(stop)))  + '\n')

		# we don't have this- find out how to get this.
		if utt_id in ivec_norm_dict_nospk:
			count+=1
			target_id = get_neighbor(sess_id, ivec_norm_dict_nospk, utt_id)
			target_sess_id = '_'.join(target_id.split('_')[:3])
			strt = int(int(target_id.split('_')[-1].split('-')[0])/10)
			stp = int((int(target_id.split('_')[-1].split('-')[1])-1)/10)
			target_csv_file_name = raw_dir + target_sess_id + '.csv'
			target_raw_feat = pd.read_csv(target_csv_file_name, dtype=np.float32)
			csv_feat_target = target_raw_feat.values.copy()
			featdata_target = np.copy(csv_feat_target)
			target_raw_norm_feat = normalizefeats(featdata_target, norm)
			sample_index_target = list(map(int,list((featdata_target[:,0]))))
			s3_list = [sample_index_target[strt:stp]]
			target_whole_func_feat = final_feat_calculate(s3_list, target_raw_norm_feat, all_raw_feat_dim)
			whole_func_feat3 = np.vstack([whole_func_feat3, target_whole_func_feat]) if whole_func_feat3.size else target_whole_func_feat
			found.append(i)
		# print(str(i)+'-th turn done')

	# if seg:
	# 	segf.close()
	print("matched: " + str(count)  + " all: " + str(len(s2_list)))

# append samples
if triplet:
	whole_func_feat1 = whole_func_feat1[found,:]
	whole_func_feat2 = whole_func_feat2[found,:]
	whole_func_feat = np.hstack([whole_func_feat1,whole_func_feat2, whole_func_feat3])
else:
	whole_func_feat = np.hstack([whole_func_feat1,whole_func_feat2])

##-----------------------------------------------------------------------
## normalization at whole session level, using scikit learn
## -- for each feature 0 mean and 1 variance
##-----------------------------------------------------------------------
# norm_whole_func_feat = preprocessing.scale(whole_func_feat)
# write to csv file

if writing==True:
	if triplet:
		feat_csv_file_name = OUTPUT_path + '/' + basename(csv_file_name).split('.csv')[0] + '_triplet_feat.csv'
	else:
		feat_csv_file_name = OUTPUT_path + '/' + basename(csv_file_name).split('.csv')[0] + '_pairfeat.csv'

	print(feat_csv_file_name)
	np.savetxt(feat_csv_file_name, whole_func_feat, delimiter=',')
