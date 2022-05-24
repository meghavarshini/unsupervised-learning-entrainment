#!/usr/bin/env bash
set -e
set -u #undefined variables will cause an exit with error
cmddir=.

export ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd)"
echo $ROOT

featdir=$ROOT/output/feats
raw_featdir=$ROOT/output/raw_feats
#audiodirroot=/media/mule/projects/ldc/Fisher_sample/fisher_eng_tr_sp_LDC2004S13_zip #For Kraken
audiodirroot=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_sample/fisher_eng_tr_sp_LDC2004S13_zip
#audiodirroot=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2
featextractfile=$ROOT/models/NED/feat_extract_nopre.py
opensmileconfig=$ROOT/models/NED/emobase2010_revised.conf

numParallelJobs=24
ctr=1

for dir in $audiodirroot/f*; do
    if [[ -d "$dir" ]]; then
      echo "$dir exists on your filesystem."
      echo "Directory: " $dir
    for f in  $dir/audio/*/*.sph; do
      echo "Processing file: " $f;
    #will work if virtual env is active
     python $featextractfile \
     --audio_file $f \
     --openSMILE_config $opensmileconfig \
     --output_path $featdir &

    if [ $(($ctr % $numParallelJobs)) -eq 0 ]
    then
      echo "Running $numParallelJobs jobs in parallel.."
      wait
    fi

	ctr=`expr $ctr + 1`
	 	
  done;
  fi
done
