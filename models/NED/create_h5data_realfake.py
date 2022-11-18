"""Code for creating fake sessions as a veritifcation step to judge whether the proposed measures are suitable for capturing the target behaviors """

from os import path
from glob import glob
import h5py
import argparse
import random
import numpy as np
import time
import os
import math


#from entrainment.config import *
#import h5py
#SEED=448
#frac_train = 0.8
#frac_val = 0.1



## Create h5 files


def make_argument_parser():
    parser = argparse.ArgumentParser(
            description  = "Processing filepaths and values required for setup")
    # input dir
    parser.add_argument("--features_dir",
            default = "/home/nasir/data/Fisher/feats_nonorm",
            # ToDo: checking any difference between "feats_nonorm" and "feats"
            help  = "features directory")
    # output dir (should be changed depending on your needs)
    parser.add_argument("--h5_directory",
            default="/home/tomcat/entrainment/NED_files/test_1_h5",
            # This is not a baseline. This is just a verification process
            help  = "directory for storing h5 files")
    return parser

# the function "clean_feat" is not needed here
def split_files(feats_dir, sess_List=None):
    sess_files = path.isfile(sess_List)
    if sess_files == 1:
        print("list of transcripts exists")
	with open(sess_List, 'r') as f:
            temp = f.read().splitlines()
            print(len(temp), len(sorted(glob(feats_dir + '/*.csv'))))
            if len(temp) == len(sorted(glob(feats_dir + '/*.csv'))):
                print("list of shuffled files exists, importing...")
		sessList = temp
		print(sessList)
            else:
		print("error in importing files")
		raise ValueError("sessList.txt is not an accurate list of files")
    else:
        print("list of transcripts does not exist")
        sessList = sorted(glob(feats_dir + '/*.csv'))
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
    sessVal = sessList[num_files_train:num_files_val+num_files_train]
    sessTest = sessList[num_files_val+num_files_train:]
    print(len(sessTrain) + len(sessVal) + len(sessTest))
    return(sessTrain, sessVal, sessTest)

# ToDo: making dataset_id and norm_id as global variables for a better coding
### Create Train Data file ###
def create_train(sessTrain, h5_dir):
    dataset_id = 'Fisher'
    norm_id = 'nonorm'
    
    X_train = np.array([])
    X_train = np.empty(shape=(0,0), dtype='float64')

    for sess_file in sessTrain:
        df_i = pd.read_csv(sess_file)
        xx = np.array(df_i)
        X_train = np.vstack([X_train, xx]) if X_train.size else xx

    X_train = X_train.astype('float64')
    hf = h5py.File(h5_dir + '/train_' + dataset_id + '_' + norm_id + '.h5', 'w')
    hf.create_dataset('dataset', data = X_train)
    hf.create_dataset('prosset', data = X_train[:,:24])
    hf.create_dataset('specset', data = X_train[:,24:150])
    hf.create_dataset('vqset', data = X_train[:,150:])
    hf.close()

    print('h5 data written to disk! Writing Train takes ' +"{0:.2f}".format(time.time() - start))
    return None

### Create Val Data file ###
def create_val(sessVal, h5_dir):
    dataset_id = 'Fisher'
    norm_id = 'nonorm'
    X_val =np.array([])
    for sess_file in sessVal:
        df_i = pd.read_csv(sess_file)
        xx=np.array(df_i)
        X_val=np.vstack([X_val, xx]) if X_val.size else xx

    X_val = X_val.astype('float64')
    hf = h5py.File(h5_dir + '/val_' + dataset_id + '_' + norm_id + '.h5', 'w')
    hf.create_dataset('dataset', data=X_val)
    hf.create_dataset('prosset', data=X_val[:,:24])
    hf.create_dataset('specset', data=X_val[:,24:150])
    hf.create_dataset('vqset', data=X_val[:,150:])
    hf.close()
    print('h5 data written to disk! Writing Val takes ' +"{0:.2f}".format(time.time() - start))
    return None

### Create Test Data file ###
def create_test(sessTest, h5_dir):
    dataset_id = 'Fisher'
    norm_id = 'nonorm'

    spk_base = 1
    X_test =np.array([])

for sess_file in sessTest:
    df_i = pd.read_csv(sess_file)
    xx=np.array(df_i)
    N = xx.shape[0]
    if np.mod(N,2)==0:
        spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
    else:
        spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
        spk_label = np.append(spk_label, spk_base)
	
    xx = np.hstack((xx, spk_label.T.reshape([N,1])))
    X_test=np.vstack([X_test, xx]) if X_test.size else xx
    spk_base += 1


    X_test = X_test.astype('float64')
    hf = h5py.File(h5_dir + '/test_' + dataset_id + '_' + norm_id + '.h5', 'w')
    hf.create_dataset('dataset', data=X_test)
    hf.create_dataset('prosset', data=X_test[:,:24])
    hf.create_dataset('specset', data=X_test[:,24:150])
    hf.create_dataset('vqset', data=X_test[:,150:])
    hf.close()
    
    print('h5 data written to disk! Writing Test takes ' +"{0:.2f}".format(time.time() - start))
    return None

if __name__ = "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    SEED = 448
    frac_train = 0.8
    frac_val = 0.1

    tr, v, te = split_files(feats_dir = args.features_dir, sess_List = args.h5_directory+"/sessList.txt")
    
    create_train(sessTrain = tr, h5_dir= args.h5_directory)
    create_val(sessVal= v, h5_dir= args.h5_directory)
    create_test(sessTest= te, h5_dir= args.h5_directory)

# feats_nonorm_nopre
#sessList= sorted(glob.glob(data_dir + '/*.csv'))
#random.seed(SEED)
#random.shuffle(sessList)

#num_files_all = len(sessList)
#num_files_train = int(np.ceil((frac_train*num_files_all)))
#num_files_val = int(np.ceil((frac_val*num_files_all)))
#num_files_test = num_files_all - num_files_train - num_files_val

#sessTrain = sessList[:num_files_train]
#sessVal = sessList[num_files_train:num_files_val+num_files_train]
#sessTest = sessList[num_files_val+num_files_train:]
#print(len(sessTrain) + len(sessVal) + len(sessTest))

## Create Train Data file

#X_train =np.array([])
#X_train = np.empty(shape=(0, 0), dtype='float64' )
#for sess_file in sessTrain:
#	df_i = pd.read_csv(sess_file)
#	xx=np.array(df_i)
#	X_train=np.vstack([X_train, xx]) if X_train.size else xx


#X_train = X_train.astype('float64')
#hf = h5py.File('data/train_Fisher_nonorm.h5', 'w')
#hf.create_dataset('dataset', data=X_train)
#hf.create_dataset('prosset', data=X_train[:,:24])
#hf.create_dataset('specset', data=X_train[:,24:150])
#hf.create_dataset('vqset', data=X_train[:,150:])
#hf.close()


## Create Val Data file

# X_val =np.array([])
# for sess_file in sessVal:
# 	df_i = pd.read_csv(sess_file)
# 	xx=np.array(df_i)
# 	X_val=np.vstack([X_val, xx]) if X_val.size else xx
#
# X_val = X_val.astype('float64')
# hf = h5py.File('data/val_Fisher_nonorm.h5', 'w')
# hf.create_dataset('dataset', data=X_val)
# hf.create_dataset('prosset', data=X_val[:,:24])
# hf.create_dataset('specset', data=X_val[:,24:150])
# hf.create_dataset('vqset', data=X_val[:,150:])
# hf.close()




## Create Test Data file
# spk_base = 1
# X_test =np.array([])
# for sess_file in sessTest:
# 	df_i = pd.read_csv(sess_file)
# 	xx=np.array(df_i)
# 	N = xx.shape[0]
# 	if np.mod(N,2)==0:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 	else:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 		spk_label = np.append(spk_label, spk_base)
# 	xx = np.hstack((xx, spk_label.T.reshape([N,1])))
# 	X_test=np.vstack([X_test, xx]) if X_test.size else xx
# 	spk_base += 1
#
#
# X_test = X_test.astype('float64')
# hf = h5py.File('data/test_Fisher_nonorm.h5', 'w')
# hf.create_dataset('dataset', data=X_test)
# hf.create_dataset('prosset', data=X_test[:,:24])
# hf.create_dataset('specset', data=X_test[:,24:150])
# hf.create_dataset('vqset', data=X_test[:,150:])
# hf.close()




# # data_dir = '~/Downloads/Fisher_corpus/feats_nonorm_nopre/'
# data_dir = out_dir
#
# sessList= sorted(glob.glob(data_dir + '/*.csv'))
# random.seed(SEED)
# random.shuffle(sessList)
#
# num_files_all = len(sessList)
# num_files_train = int(np.ceil((frac_train*num_files_all)))
# num_files_val = int(np.ceil((frac_val*num_files_all)))
# num_files_test = num_files_all - num_files_train - num_files_val
#
# sessTrain = sessList[:num_files_train]
# sessVal = sessList[num_files_train:num_files_val+num_files_train]
# sessTest = sessList[num_files_val+num_files_train:]
# print(len(sessTrain) + len(sessVal) + len(sessTest))

## Create Train Data file

# X_train =np.array([])
# X_train = np.empty(shape=(0, 0), dtype='float64' )
# for sess_file in sessTrain:
# 	df_i = pd.read_csv(sess_file)
# 	xx=np.array(df_i)
# 	X_train=np.vstack([X_train, xx]) if X_train.size else xx
#
#
# X_train = X_train.astype('float64')
# hf = h5py.File('data/train_Fisher_nonorm_nopre.h5', 'w')
# hf.create_dataset('dataset', data=X_train)
# hf.create_dataset('prosset', data=X_train[:,:24])
# hf.create_dataset('specset', data=X_train[:,24:150])
# hf.create_dataset('vqset', data=X_train[:,150:])
# hf.close()


## Create Val Data file
#
# X_val =np.array([])
# for sess_file in sessVal:
# 	df_i = pd.read_csv(sess_file)
# 	xx=np.array(df_i)
# 	X_val=np.vstack([X_val, xx]) if X_val.size else xx
#
# X_val = X_val.astype('float64')
# hf = h5py.File('data/val_Fisher_nonorm_nopre.h5', 'w')
# hf.create_dataset('dataset', data=X_val)
# hf.create_dataset('prosset', data=X_val[:,:24])
# hf.create_dataset('specset', data=X_val[:,24:150])
# hf.create_dataset('vqset', data=X_val[:,150:])
# hf.close()




## Create Test Data file
# spk_base = 1
# X_test =np.array([])
# for sess_file in sessTest:
# 	df_i = pd.read_csv(sess_file)
# 	xx=np.array(df_i)
# 	N = xx.shape[0]
# 	if np.mod(N,2)==0:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 	else:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 		spk_label = np.append(spk_label, spk_base)
# 	xx = np.hstack((xx, spk_label.T.reshape([N,1])))
# 	X_test=np.vstack([X_test, xx]) if X_test.size else xx
# 	spk_base += 1
#
#
# X_test = X_test.astype('float64')
# hf = h5py.File('data/test_Fisher_nonorm_nopre.h5', 'w')
# hf.create_dataset('dataset', data=X_test)
# hf.create_dataset('prosset', data=X_test[:,:24])
# hf.create_dataset('specset', data=X_test[:,24:150])
# hf.create_dataset('vqset', data=X_test[:,150:])
# hf.close()
