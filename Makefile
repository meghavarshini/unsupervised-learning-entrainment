CORPUS=/media/mule/projects/ldc/Fisher_sample
AUDIO_DIR_ROOT=$(CORPUS)/fisher_eng_tr_sp_LDC2004S13_zip

# Path to OpenSMILE config
OPENSMILE_CONFIG=feats/emobase2010_haoqi_revised.conf

# Directory with all transcripts
TRANSCRIPT_DIR=$(CORPUS)/fe_03_p1_tran/data/trans/all_trans

SPH_FILES:= $(shell find $(AUDIO_DIR_ROOT) -type f -name '*.sph')
WAV_FILES:= $(patsubst $(AUDIO_DIR_ROOT)%, build%, $(SPH_FILES:.sph=.wav))
RAW_CSV_FILES:= $(WAV_FILES:.wav=_features_raw.csv)
NORMED_CSV_FILES:= $(WAV_FILES:.wav=_features_normed.csv)


all: $(NORMED_CSV_FILES)

# Location of the sph2wav script
SPH2WAV=scripts/sph2wav


# Recipe to convert .sph files to .wav files
build/%.wav: $(SPH2WAV) $(AUDIO_DIR_ROOT)/%.sph
	@mkdir -p $(@D)
	$^ $@

build/%_features_raw.csv: build/%.wav
	SMILExtract -C $(OPENSMILE_CONFIG) -I $< -O $@

# We define the special target .SECONDEXPANSION in order to handle expansions
# in prerequisites ($$).
.SECONDEXPANSION:
build/%_features_normed.csv: feats/feat_extract_nopre.py\
							 build/%_features_raw.csv\
							 $(TRANSCRIPT_DIR)/$$(notdir %).txt
	python $^ $@ 
