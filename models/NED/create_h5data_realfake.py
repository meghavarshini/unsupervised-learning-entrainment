#!/usr/bin/env python

from glob import glob
from argparse import ArgumentParser
import h5py
from typing import Dict


def make_argument_parser():
    parser = ArgumentParser(description="Create real and fake data")
    parser.add_argument("input_files", help="input files", nargs="+")
    return parser

def create_sessions_dict(data_dir: str):

    SEED = 448
    frac_train = 0.8
    frac_val = 0.1
    session_list = sorted(glob(data_dir + "/*.csv"))

    # Set random seed
    random.seed(SEED)
    random.shuffle(session_list)

    num_files_all = len(session_list)
    num_files = {
        "train": int(np.ceil((frac_train * num_files_all))),
        "val": int(np.ceil((frac_val * num_files_all))),
    }
    num_files["test"]: num_files_all - num_files["train"] - num_files["val"]

    sessions = {
        "train": session_list[: num_files["train"]],
        "val": session_list[
            num_files["train"] : num_files["val"] + num_files["train"]
        ],
        "test": session_list[num_files["val"] + num_files["train"] :],
    }
    print(sum(len(x) for x in sessions.values()))
    return sessions


def create_h5_file(X, split: str, dataset_type: str):
    """Save sub-arrays to h5 file"""
    X = X.astype("float64")
    hf = h5py.File(f"data/{split}_Fisher_{dataset_type}.h5", "w")
    hf.create_dataset("dataset", data=X)
    hf.create_dataset("prosset", data=X[:, :24])
    hf.create_dataset("specset", data=X[:, 24:150])
    hf.create_dataset("vqset", data=X[:, 150:])
    hf.close()


def create_data_file(sessions: Dict, split: str, dataset_type: str):
    """Create a data file.
    split should be either 'train', 'test', or 'val'
    dataset_type should be either 'nonorm' or 'nonorm_nopre'
    """
    X = np.array([])
    X = np.empty(shape=(0, 0), dtype="float64")
    for session_file in sessions[dataset_type]:
        df = pd.read_csv(session_file)
        df_as_array = np.array(df)
        X = np.vstack([X, df_as_array]) if X.size else df_as_array

    create_h5_file(X, split, dataset_type)


def create_test_file(sessions: Dict, dataset_type: str):
    """Create test data file"""
    spk_base = 1
    X = np.array([])
    for session_file in sessions["test"]:
        df = pd.read_csv(session_file)
        xx = np.array(df)
        N = xx.shape[0]
        # For testing, we need to repeat array of openSMILE features
        if np.mod(N, 2) == 0:
            spk_label = np.tile([spk_base, spk_base + 1], [1, N / 2])
        else:
            spk_label = np.tile([spk_base, spk_base + 1], [1, N / 2])
            spk_label = np.append(spk_label, spk_base)
        xx = np.hstack((xx, spk_label.T.reshape([N, 1])))
        X = np.vstack([X, xx]) if X.size else xx
        spk_base += 1

    create_h5_file(X, "test", dataset_type)


def create_data_files(sessions, dataset_type: str):
    create_data_file(sessions, "train", dataset_type)
    create_data_file(sessions, "val", dataset_type)
    create_test_data_file(sessions, dataset_type)


if __name__ == "__main__":
    parser = make_argument_parser()
    parser.parse_args()

    ## Create h5 files
    sessions = create_sessions_dict(args.data_dir)
    create_data_files(sessions, "nonorm")
    # # data_dir = '~/Downloads/Fisher_corpus/feats_nonorm_nopre/'
    # create_data_files(sessions, "nonorm_nopre")
