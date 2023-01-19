import argparse
from glob import glob
import random
from feats.create_h5data import create_test
from feats.test import load_h5
# from feats.test import model_testing



def make_argument_parser():
	parser = argparse.ArgumentParser(
        				description="Processing filepaths and values required for setup")
	parser.add_argument("--h5_directory",
						default="/Users/meghavarshinikrishnaswamy/transcripts/files_for_pipeline/output",
						help= "directory for storing h5 files")
	return parser

def shuffle_files(input_dir):
    sessList = sorted(glob(input_dir + '/*.csv'))
    print("sessList: ", sessList)
    print("creating a list of shuffled feature files...")
    # print("sessList", sessList)
    random.shuffle(sessList)
    return sessList

if __name__ == "__main__":

    parser = make_argument_parser()
    args = parser.parse_args()

    #Create h5 file
    sessTest = shuffle_files(args.h5_directory)
    create_test(sessTest=sessTest, h5_dir=args.h5_directory)
    test_h5 = args.h5_directory +'/test_Fisher_nonorm.h5'

    test_input = load_h5(test_h5)