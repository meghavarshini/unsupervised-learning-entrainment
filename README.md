------------------------------------------------------------------------------------------
Software supporting "Modeling Vocal Entrainment in Conversational Speech using Deep Unsupervised Learning"
------------------------------------------------------------------------------------------

original code by Md Nasir 
Modified for Python3 by Megh Krishnaswamy, Adarsh Pyrelal, and John Culnan.


#### Updated on January 17, 2024 (Work in Progress)

#### Scripts are written in Python 3.11


------------
Dependencies and Requirements
------------
1. Bash
2. [OpenSMILE](https://github.com/audeering/opensmile)
3. [sph2pipe](https://github.com/burrmill/sph2pipe)
5. [Kaldi](https://kaldi-asr.org)
4. LDC corpus and metadata files <br>(for ToMCAT users, a small sample of the corpus is available at `kraken.sista.arizona.edu:/media/mule/projects/ldc`)
<br> note: move all files in the

------------------------
Summary of directories and files
--------------------------------

- feat/ :- Directory containing scripts for acoustic feature extraction and functional computation
- models/ :- Directory containing scripts for different deep unsupervised learning models for entrainment
- utils/ :- Directory containing utility files used by other scripts

------------------------
Files that need to be edited by user to add filepaths:
------------------------

- `./Makefile`

------------------------
Start Point:
------------------------
1. Setup and activate virtual environment/ conda environment
2. Run `pip install -e .`
3. To run the code on your system, download and set-up the LDC data, and access/create `Fisher_meta.csv`
    -   `scp -r [username]@kraken.sista.arizona.edu:/media/mule/projects/ldc [local directory]`
3. Edit ./Makefile with relevant filepaths. These are listed in the 'Parameters' section.
4. Run `bash ./Makefile` to extract OpenSMILE features for all sound files.
5. Run ` python feats/feat_extract_nopre.py [all optional arguments]` to extract the feature set.
    1. Each line contains time- and speaker- and utterance- aligned features for every pair of turns spoken by different speakers.
    2. Do not edit the script, instead, change the input parameters while running the script.
    3. This will require feature files created in the previous step.
6. Run `python feats/create_h5data.py --features_dir [extracted feature directory]`. This will create h5 files with the training, dev and test sets.
7. Run `python train.py`. This will create a `.pt` file using the h5 files.
8. Run `python test.py` 30 times, and write the std output to a file. This will need the trained model created in the previous step, as well as `test.h5`. This creates the real and fake datasets, and outputs a score for each run.

------------------------
Permissions:
------------------------
ToDo- edit this to reflect the new files
Make sure the following directories/files have permissions:
1. chmod 777 feats
2. chmod 777 model/NED
3. chmod 777 feats/emobase2010_haoqi_revised.conf
4. chmod 777 models/NED/emobase2010_revised.conf
