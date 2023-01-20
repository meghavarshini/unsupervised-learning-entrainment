import argparse
from feats.test import load_h5


# from feats.test import model_testing

def make_argument_parser():
    parser = argparse.ArgumentParser(
        description="Processing filepaths and values required for setup")
    parser.add_argument("--h5_directory",
                        default="/home/tomcat/entrainment/asist3",
                        help="directory for storing h5 files")
    return parser


def load_h5(file):
    print("loading h5 file: ", file)
    file = h5py.File(file, 'r')
    test = np.array(file['dataset'])
    print("loading complete!")
    return test


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    test_h5 = args.h5_directory + '/test_Fisher_nonorm.h5'

    test_input = load_h5(test_h5)
