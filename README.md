------------------------------------------------------------------------------------------
Software supporting "Modeling Vocal Entrainment in Conversational Speech using Deep Unsupervised Learning"
------------------------------------------------------------------------------------------

original code by Md Nasir 
Modified for Python3 by Megh Krishnaswamy and Adarsh Pyrelal


#### Updated on June 3, 2022 (Work in Progress)
#### Scripts are written in Python 3.9


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
--------------------------------
ToDo- edit this with latest files
- ~~`entrainment_config.py`
- ~~`feats/run_all_nopre.sh`
- ~~`models/NED/run_all_nopre.sh`

------------------------
Start Point:
------------------------
ToDo- edit this for instructions for the makefile

1. Setup and activate virtual environment
2. Run `pip install -e .`
1. To run the code on your system, download and set-up the LDC data, and access/create `Fisher_meta.csv`
    -   `scp -r [username]@kraken.sista.arizona.edu:/media/mule/projects/ldc [local directory]`
1. Add a step for setup.py, wheel and installing requirements   
2. ~~Ensure you have installed all required python libraries
3. ~~Edit `entrainment_config.py`, `feats/run_all_nopre.sh` and `models/NED/run_all_nopre.sh` with relevant filepaths
4. ~~Run `feats/run_all_nopre.sh` and `models/NED/run_all_nopre.sh` to extract OpenSMILE features for all sound files
5. Run `models/triplet/prep/create_kaldi_files.py` to extract Kaldi files
6. Open feats directory to follow the steps for modelling

------------------------
Permissions:
------------------------
ToDo- edit this to reflect the new files
Make sure the following directories/files have permissions:
1. chmod 777 feats
2. chmod 777 model/NED
3. chmod 777 feats/emobase2010_haoqi_revised.conf
4. chmod 777 models/NED/emobase2010_revised.conf
~~3. chmod 755 feats/run_all_nopre.sh
~4. chmod 755 models/NED/run_all_nopre.sh
