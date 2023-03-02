import create_h5data
import test

def make_argument_parser():
    parser = argparse.ArgumentParser(
            description  = "Processing filepaths and values required for setup")
    # input dir
    parser.add_argument("--features_dir",
            default = "/home/tomcat/entrainment/feat_files/baseline_1_feats",
            help  = "features directory")
    # output dir (should be changed depending on your needs)
    parser.add_argument("--h5_directory",
            default="/home/tomcat/entrainment/NED_files/baseline_new_h5",
            help  = "directory for storing h5 files")
    parser.add_argument('--no-cuda', action='store_true',
                        default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--hff', default="/home/tomcat/entrainment/NED_files/mini/test_Fisher_acoustic_nonorm.h5",
                        help='location of h5 file with the test data')
    parser.add_argument('--model_name',
                        default="/home/tomcat/entrainment/NED_files/mini/models/trained_Fisher_acoustic_nonorm_l1_30dim.pt",
                        help='name associated with the trained model')

    return parser

if __name__ == "__main__":
# Setup argparse and basic variables
    os.getcwd()
    parser = make_argument_parser()
    args = parser.parse_args()

# Get list of feature files not used in the mini corpus's Train, Test or Dev partitions,
# shuffle filenames and write to a test file
    os.system('diff -r /home/tomcat/entrainment/feat_files/baseline_1_feats \
     /home/tomcat/entrainment/feat_files/mini_fisher_corpus | grep csv|shuf $1  |\
      awk '{print "/home/tomcat/entrainment/feat_files/baseline_1_feats/" $4}' >\
       /home/tomcat/entrainment/feat_files/baseline_mini_h5/dev2list.txt')

    dev2list = "/home/tomcat/entrainment/feat_files/baseline_mini_h5/dev2list.txt"

#Open file list, read required number of files (1760), and remove the test file
    with open(dev2list, 'r') as f:
    	dev2 = f.read().splitlines()[:1759]
    os.system('rm /home/tomcat/entrainment/feat_files/baseline_mini_h5/dev2list.txt')

#Create Partition
    create_test(sessTest=dev2, h5_dir=args.h5_directory, data_type= "dev2")

# Setup requirements and run test.py
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	if not args.no_cuda and torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	X_test1 = load_h5(args.hff)
	test_run = model_testing(args.model_name, X_test1, args.cuda)

