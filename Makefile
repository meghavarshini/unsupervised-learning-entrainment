# Makefile to automate entrainment detection analysis
# Change the values in the 'Parameters' block if you need to

# ----------
# Parameters
# ----------
# Location of the corpus (transcripts + sound files)
CORPUS=/media/mule/projects/ldc/Fisher_sample

# Location of the corpus sound files 
AUDIO_DIR_ROOT=$(CORPUS)/fisher_eng_tr_sp_LDC2004S13_zip

# Path to OpenSMILE configs
OPENSMILE_CONFIG_BASELINE=feats/emobase2010_haoqi_revised.conf
OPENSMILE_CONFIG_NED=models/NED/emobase2010_mod.conf

# Directory with all transcripts
TRANSCRIPT_DIR=$(CORPUS)/fe_03_p1_tran/data/trans/all_trans

# -------------------------------------------------------------------

# Find all .sph files recursively inside AUDIO_DIR_ROOT
SPH_FILES:= $(shell find $(AUDIO_DIR_ROOT) -type f -name '*.sph')

# Replace .sph suffix with .wav suffix, then replace AUDIO_DIR_ROOT with build
# (output files will go into the build folder).
WAV_FILES:= $(patsubst $(AUDIO_DIR_ROOT)%, build%, $(SPH_FILES:.sph=.wav))

# Construct file names for raw CSV files
BASELINE_RAW_CSV_FILES:= $(WAV_FILES:.wav=_features_raw_baseline.csv)

NED_RAW_CSV_FILES:= $(WAV_FILES:.wav=_features_raw_ned.csv)

# Construct file names for normed CSV files. 
BASELINE_NORMED_CSV_FILES:= $(WAV_FILES:.wav=_features_normed_baseline.csv)
NED_NORMED_CSV_FILES:= $(WAV_FILES:.wav=_features_normed_ned.csv)

all: $(NED_NORMED_CSV_FILES) $(BASELINE_NORMED_CSV_FILES)

# Recipe to convert .sph files to .wav files
build/%.wav: scripts/sph2wav $(AUDIO_DIR_ROOT)/%.sph
	@mkdir -p $(@D)
	$^ $@

build/%_features_raw_baseline.csv: build/%.wav
	SMILExtract -C $(OPENSMILE_CONFIG_BASELINE) -I $< -O $@

build/%_features_raw_ned.csv: build/%.wav
	SMILExtract -C $(OPENSMILE_CONFIG_NED) -I $< -O $@

# We define the special target .SECONDEXPANSION in order to handle expansions
# in prerequisites ($$).
.SECONDEXPANSION:
build/%_features_normed_baseline.csv: feats/feat_extract_nopre.py\
							 build/%_features_raw_baseline.csv\
							 $(TRANSCRIPT_DIR)/$$(notdir %).txt
	$^ $@

# We define the special target .SECONDEXPANSION in order to handle expansions
# in prerequisites ($$).
.SECONDEXPANSION:
build/%_features_normed_ned.csv: feats/feat_extract_nopre.py\
							 build/%_features_raw_ned.csv\
							 $(TRANSCRIPT_DIR)/$$(notdir %).txt
	$^ $@
