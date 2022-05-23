#!/usr/bin/env bash
set -e
set -u #undefined variables will cause an exit with error

# cmddir=/home/nasir/inter_dynamics/scripts/NPC
# replaced with $PWD

featdir=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/feats
raw_featdir=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/raw_feats
audiodirroot=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_sample
#audiodirroot=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2
featextractfile=/Users/meghavarshinikrishnaswamy/github/unsupervised-learning-entrainment/feats/feat_extract_nopre.py
opensmileconfig=/Users/meghavarshinikrishnaswamy/github/unsupervised-learning-entrainment/feats/emobase2010_haoqi_revised.conf

numParallelJobs=24
ctr=1
# for dir in $audiodir/*;
# do  
# 	cd $dir
# 	for file in *;
# 	do 
# 	python $cmddir/feat_extract.py --audio_file $dir/$file --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
# done
# 	cd ..
# done

# for file in $audiodir/*.csv;
# do
# 	python $cmddir/feat_extract_nopre.py --audio_file $file --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
# done

##make this recursive, search all subfolders
for dir in $audiodirroot/f*; do
  if [[ -d "$dir" ]]
    then
    echo "$dir exists on your filesystem."
    echo flag2 $dir
	  for f in  $dir/audio/*/*.sph; do
		  echo flag1 $f;
	    (
	 	  python $featextractfile --audio_file $f --openSMILE_config $opensmileconfig --output_path $featdir #will work if virtual env is active
	 	  ) &

	if [ $(($ctr % $numParallelJobs)) -eq 0 ]
	then
#		echo "Running $numParallelJobs jobs in parallel.."
		wait
	fi
	ctr=`expr $ctr + 1`
	done;
	fi
done
