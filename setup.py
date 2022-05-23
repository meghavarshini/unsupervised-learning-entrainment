""" setuptools-based setup module. """

from setuptools import setup, find_packages
from setuptools import setup

import argparse
from argparse import Namespace
import os
from os.path import basename
from os.path import exists
import sys
import csv
import pandas as pd
import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt
import pdb
import glob
import random
import h5py
import kaldi_io
# from aeent import *
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import pprint, pickle
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from scipy import spatial

setup(
    name="unsupervised-learning-entrainment",
    description="Deep learning model for vocal entrainment",
    url="https://github.com/meghavarshini/unsupervised-learning-entrainment",
    packages = find_packages(),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Intended Audience :: Science/Research",
    #     "Topic :: Scientific/Engineering :: Artificial Intelligence",
    # ],
    keywords="vocalic feature modelling",
    # zip_safe=False,
    install_requires=[
        "wheel",
        "torch==1.11.0",
        "torchvision==0.12.0",
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "tqdm",
        "webvtt-py",
        "transformers",
        "h5py",
        "kaldi-io==0.9.4",
        "scipy"
    ],
    # extras_require={"mmc_server": ["uvicorn", "fastapi"]},
    python_requires=">=3.9",
)