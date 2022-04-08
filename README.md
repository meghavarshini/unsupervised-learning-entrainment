------------------------------------------------------------------------------------------
Software supporting "Modeling Vocal Entrainment in Conversational Speech using Deep Unsupervised Learning"
------------------------------------------------------------------------------------------

written by Md Nasir (modified for Python3 by Megh Krishnaswamy)


#### Updated on Apr 8, 2022 (Work in Progress)
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

- `entrainment_config.py`
- `feats/run_all_nopre.sh`
- `models/NED/run_all_nopre.sh`

------------------------
Start Point:
--------------------------------
1. Download and set-up the LDC data, and access/create `Fisher_meta.csv`
2. Ensure you have installed all dependencies
3. Edit `entrainment_config.py`, `feats/run_all_nopre.sh` and `models/NED/run_all_nopre.sh` with relevant filepaths
4. Run `feats/run_all_nopre.sh` or `models/NED/run_all_nopre.sh` to extract OpenSMILE features for all sound files
5. Run `models/triplet/prep/create_kaldi_files.py` to extract Kaldi files
			
