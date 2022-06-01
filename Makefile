CORPUS=/media/mule/projects/ldc/Fisher_sample
AUDIO_DIR_ROOT=$(CORPUS)/fisher_eng_tr_sp_LDC2004S13_zip
OPENSMILE_CONFIG=feats/emobase2010_haoqi_revised.conf
# File path of the directory with all transcripts
TRANSCRIPT_DIR=$(CORPUS)/fe_03_p1_tran/data/trans/all_trans

SPH_FILES:= $(shell find $(AUDIO_DIR_ROOT) -type f -name '*.sph')
WAV_FILES:= $(patsubst $(AUDIO_DIR_ROOT)%, build%, $(SPH_FILES:.sph=.wav))
CSV_FILES:= $(WAV_FILES:.wav=_features.csv)

# Location of the sph2wav script
SPH2WAV=scripts/sph2wav

# Recipe to convert .sph files to .wav files
# The .INTERMEDIATE annotation makes it so that the .wav files are deleted when
# they are not needed any longer.
.INTERMEDIATE:
build/%.wav: $(SPH2WAV) $(AUDIO_DIR_ROOT)/%.sph
	@mkdir -p $(@D)
	$^ $@

build/%_features_raw.csv: build/%.wav
	SMILExtract -C $(OPENSMILE_CONFIG) -I $< -O $@

build/%_features_normed.csv: feats/feat_extract_nopre.py build/%_features_raw.csv $(TRANSCRIPT_DIR)/%.txt
	python $^ $<

all: $(firstword $(CSV_FILES))
