#!/usr/bin/env python

from __future__ import print_function

from kaldi.alignment import NnetAligner
from kaldi.fstext import SymbolTablecd
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

# Construct aligner
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
# aligner = NnetAligner.from_files("final.mdl", "tree", "L.fst", "words.txt",
#                                  "disambig.int", decodable_opts=decodable_opts)

# Define feature pipelines as Kaldi rspecifiers
feats_rspec = "ark:compute-mfcc-feats --config=mfcc.conf scp:./data/wav.scp ark:- |"
ivectors_rspec = (
    "ark:compute-mfcc-feats --config=mfcc.conf scp:./data/wav.scp ark:-"
    " | ivector-extract-online2 --config=ivector.conf ark:./data/spk2utt ark:- ark:- |"
    )
# ivector-extract-online2: Kaldi function
#compute-mfcc-feats
# Align wav files
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i, open("text") as t:
    for (fkey, feats), (ikey, ivectors), line in zip(f, i, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == ikey == tkey)