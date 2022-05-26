#!/usr/bin/env bash
set -e
set -u #undefined variables will cause an exit with error
cmddir=.

export ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd)"
echo "Root: " $ROOT

corpus=/media/mule/projects/ldc/Fisher_sample
#corpus=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_sample
audiodirroot=$corpus/fisher_eng_tr_sp_LDC2004S13_zip
transcript_dir=$corpus/fe_03_p1_tran/data/trans/all_trans

outdir=$ROOT/baseline_feats/feats_norm
raw_featdir=$ROOT/baseline_feats/raw_feats

mkdir -p $outdir $raw_featdir

featextractfile=$ROOT/feats/feat_extract_nopre.py
opensmileconfig=$ROOT/feats/emobase2010_haoqi_revised.conf

numParallelJobs=24
ctr=1

##make this recursive, search all subfolders
for dir in $audiodirroot/f*; do
    if [[ -d "$dir" ]]; then
        for f in  $dir/audio/*/*.sph; do
            if [[ -f $f ]]; then
                echo "Processing file: " $f;
                #will work if virtual env is active
                python $featextractfile \
                    --audio_file $f \
                    --openSMILE_config $opensmileconfig \
                    --feat_dir $raw_featdir \
                    --transcript_dir $transcript_dir\
                    --output_path $outdir  &
            fi

        if [ $(($ctr % $numParallelJobs)) -eq 0 ]; then
            echo "Running $numParallelJobs jobs in parallel.."
            wait
        fi

        ctr=`expr $ctr + 1`

        done;
    fi
done
