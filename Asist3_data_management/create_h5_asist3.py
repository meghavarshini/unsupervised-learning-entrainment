import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import math
import h5py
from glob import glob
import random

def make_argument_parser():
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup")
	parser.add_argument("--input_directory",
						default="./multicat_feats_addressee",
						help="directory for opensmile feat files (csv)")
	parser.add_argument("--h5_directory",
						default="./multicat_h5_output",
						help="directory for storing h5 files")
	return parser

### Shuffle files
def shuffle_files(input_dir):
	sessList = sorted(glob(input_dir + '/*.csv'))
	print("sessList: ", sessList)
	print("creating a list of shuffled feature files...")
	# print("sessList", sessList)
	random.shuffle(sessList)
	return sessList

#### Split files to create train, test, val sets
def split_files(input_dir, sess_List = None):
	SEED = 448
	frac_train = 0.7
	frac_val = 0.1
	
	sess_files = os.path.isfile(sess_List)
	if sess_files == 1:
		print("list of transcripts exists")
		with open(sess_List, 'r') as f:
			temp = f.read().splitlines()
			print(f'Check if no. of files match:')
			print(len(temp), len(sorted(glob(input_dir + '/*.csv'))))
			if len(temp) == len(sorted(glob(input_dir + '/*.csv'))):
				print("list of shuffled files exists, importing...")
				sessList = temp
				print(sessList)
			else:
				print("error in importing files")
				raise ValueError("sessList.txt is not an accurate list of files")
	else:
		print("list of transcripts does not exist")
		sessList = sorted(glob(input_dir + '/*.csv'))
		print("sessList: ", sessList)
		print("creating a list of shuffled feature files and saving to disk...")
		random.seed(SEED)
		random.shuffle(sessList)
		with open(sess_List, 'w') as f:
			f.writelines("%s\n" % i for i in sessList)
		with open(sess_List, 'r') as f:
			sessList = f.read().splitlines()

	num_files_all = len(sessList)
	num_files_train = int(np.ceil((frac_train * num_files_all)))
	print("num_files_train: ", num_files_train)
	num_files_val = int(np.ceil((frac_val * num_files_all)))
	print("num_files_val", num_files_val)
	num_files_test = num_files_all - num_files_train - num_files_val
	print("num_files_test", num_files_test)
	sessTrain = sessList[:num_files_train]
	sessVal = sessList[num_files_train:num_files_val + num_files_train]
	sessTest = sessList[num_files_val + num_files_train:]
	print(len(sessTrain) + len(sessVal) + len(sessTest))
	return (sessTrain, sessVal, sessTest)

### Remove nan rows
def clean_feat(XX, dim):
    ind = []
    for i, pair in enumerate(XX):
        x = pair[0:dim]
        y = pair[dim:]
        if np.any(x) and np.any(y) and (not np.any(np.isnan(x))) and (not np.any(np.isnan(y))):
            ind.append(i)
    XX = XX[ind, :]
    return XX

#### Create Training Data file ####
def create_train(sessList, h5_dir):
	dim = 228
	spk_base = 1
	X_train = np.array([])
	for sess_file in sessList:
		print("session file working on: ", sess_file)
		#load feature file as a dataframe
		df_i = pd.read_csv(sess_file, header=None)
		# load feature file as an array
		xx = np.array(df_i)
		print("no. of turn pairs: ", xx.shape[0])
		N = xx.shape[0]
		if np.mod(N, 2) == 0:
			print("even number of data rows in feature array")
			print("N/2: ", math.floor(N / 2), N / 2)
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
		else:
			print("odd number of data rows, taking modulus...")
			print("N/2: ", math.floor(N / 2))
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
			spk_label = np.append(spk_label, spk_base)
		xx = np.hstack((xx, spk_label.T.reshape([N, 1])))
		xx = clean_feat(xx, dim)
		print(sess_file, "examined and train array created")
		X_train = np.vstack([X_train, xx]) if X_train.size else xx
		spk_base += 1
		# print(spk_base)
	# print(X_test[:10])

	X_train = X_train.astype('float64')
	hf = h5py.File(h5_dir + '/train_ASIST.h5', 'w')
	hf.create_dataset('dataset', data=X_train)
	hf.create_dataset('prosset', data=X_train[:, :24])
	hf.create_dataset('specset', data=X_train[:, 24:150])
	hf.create_dataset('vqset', data=X_train[:, 150:])
	hf.close()
	return None

#### Create Val Data file ####
def create_val(sessList, h5_dir):
	spk_base = 1
	X_val = np.array([])
	for sess_file in sessList:
		print("session file working on: ", sess_file)
		#load feature file as a dataframe
		df_i = pd.read_csv(sess_file, header=None)
		# load feature file as an array
		xx = np.array(df_i)
		print("feature array shape: ", xx.shape[0])
		N = xx.shape[0]
		if np.mod(N, 2) == 0:
			print("even number of data rows in feature array")
			print("N/2: ", math.floor(N / 2), N / 2)
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
		else:
			print("odd number of data rows, taking modulus...")
			print("N/2: ", math.floor(N / 2))
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
			spk_label = np.append(spk_label, spk_base)
		xx = np.hstack((xx, spk_label.T.reshape([N, 1])))
		print(sess_file, "examined and val array created")
		X_val = np.vstack([X_val, xx]) if X_val.size else xx
		spk_base += 1
		# print(spk_base)
	# print(X_val[:10])

	X_val = X_val.astype('float64')
	hf = h5py.File(h5_dir + '/val_ASIST.h5', 'w')
	hf.create_dataset('dataset', data=X_val)
	hf.create_dataset('prosset', data=X_val[:, :24])
	hf.create_dataset('specset', data=X_val[:, 24:150])
	hf.create_dataset('vqset', data=X_val[:, 150:])
	hf.close()
	return None

#### Create Test Data file ####
def create_test(sessList, h5_dir):
	spk_base = 1
	X_test = np.array([])
	for sess_file in sessList:
		print("session file working on: ", sess_file)
		#load feature file as a dataframe
		df_i = pd.read_csv(sess_file, header=None)
		# load feature file as an array
		xx = np.array(df_i)
		print("feature array shape: ", xx.shape[0])
		N = xx.shape[0]
		if np.mod(N, 2) == 0:
			print("even number of data rows in feature array")
			print("N/2: ", math.floor(N / 2), N / 2)
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
		else:
			print("odd number of data rows, taking modulus...")
			print("N/2: ", math.floor(N / 2))
			spk_label = np.tile([spk_base, spk_base + 1], [1, math.floor(N / 2)])
			spk_label = np.append(spk_label, spk_base)
		xx = np.hstack((xx, spk_label.T.reshape([N, 1])))
		print(sess_file, "examined and test array created")
		X_test = np.vstack([X_test, xx]) if X_test.size else xx
		spk_base += 1
		# print(spk_base)
	# print(X_test[:10])

	X_test = X_test.astype('float64')
	hf = h5py.File(h5_dir + '/test_ASIST.h5', 'w')
	hf.create_dataset('dataset', data=X_test)
	hf.create_dataset('prosset', data=X_test[:, :24])
	hf.create_dataset('specset', data=X_test[:, 24:150])
	hf.create_dataset('vqset', data=X_test[:, 150:])
	hf.close()
	return None

def load_h5(file):
	print("loading h5 file: ", file)
	file = h5py.File(file, 'r')
	test = np.array(file['dataset'])
	print("loading complete!")
	return test


if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()

	input_directory = Path(args.input_directory).resolve()
	h5_output_directory = Path(args.h5_directory).resolve()
	print(f"input: {input_directory}\n output: {h5_output_directory}")
	
	#Check if output directory exists, if not, create it:   
	if not h5_output_directory.exists():
		h5_output_directory.mkdir(parents=True, exist_ok=True)
		print(f"Could not find specified output directory {h5_output_directory}. Creating directory...")
	else:
		print(f"Specified output directory {h5_output_directory} already exists. Continuing...")

	# Shuffle and create list of files for each set
	tr, v, te = split_files(input_dir=args.input_directory, sess_List=args.h5_directory + "/sessList.txt")

	# create_train(sessList=tr, h5_dir=args.h5_directory)
	# train_h5 = args.h5_directory + '/train_ASIST.h5'
	# train_input = load_h5(train_h5)
	# print(f'Train array shape: {np.shape(train_input)}')

	create_val(sessList=v, h5_dir=args.h5_directory)
	val_h5 = args.h5_directory + '/val_ASIST.h5'
	val_input = load_h5(val_h5)
	print(f'Val array shape: {np.shape(val_input)}')

	create_test(sessList=te, h5_dir=args.h5_directory)
	test_h5 = args.h5_directory + '/test_ASIST.h5'
	test_input = load_h5(test_h5)
	print(f'Test array shape: {np.shape(test_input)}')
	
	## Create h5 file: MultiCAT Test Set ONLY
	# sessList = shuffle_files(args.input_directory)
	# create_test(sessList=sessList, h5_dir=args.h5_directory)
	# test_h5 = args.h5_directory + '/test_MultiCAT.h5'
	# test_input = load_h5(test_h5)
	# print(f'Test array shape: {np.shape(test_input)}')


