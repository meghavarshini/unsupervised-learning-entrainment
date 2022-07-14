import argparse
import csv
from os import listdir
from pathlib import Path

def make_argument_parser():
	#ToDo: add kraken defaults
	parser = argparse.ArgumentParser(
		description="Processing filepaths and values required for setup"
	)
	parser.add_argument("-transcript_dir",
						default = "/media/mule/projects/ldc/fisher-corpus/fe_03_p1_tran/data/trans/all_trans",
						help="location of transcripts")
	parser.add_argument("-audio_dir",
						default = "/media/mule/projects/ldc/fisher-corpus/fisher_eng_tr_sp_LDC2004S13_zip",
						help="location of audio files")
	parser.add_argument("-meta_file",
						default = "/media/mule/projects/ldc/fisher-corpus/Fisher_meta.csv",
						help="location of metadata file")
	parser.add_argument("-sph2pipe",
						default = "/../../../scripts/sph2wav",
						help="location of sph2pipe software")
	return parser

def create_meta_dict(metafile):
	print('metafile: ', metafile)
	metaf = open(metafile, 'r')
	reader = csv.reader(metaf)
	metadata ={}
	for row in reader:
		metadata[row[0]] = row[1:]

	return metadata



def create_kaldi_files(audio_dir_root, transcript_dir, metafile, sph2pipe, uttf, segf, wavscpf):
	for dir in listdir(audio_dir_root):
		if "fisher_eng_tr" in dir:
			subdir = Path(audio_dir_root + "/" + dir + "/audio")
			print("now working on... ", subdir)
			for subsubdir in subdir.glob("*/"):
				print("now working on... ", subsubdir)
				for audio in Path(subsubdir).glob("*.sph"):
					print("audio file found... " + str(audio))
					sess_id = audio.stem.split('_')[-1]
					# sess_id = int(str(audio).split('_')[-1].split('.')[0])
					#ToDo: fix the line below to work correctly

					wavscpf.write(str(audio) + ' '+ sph2pipe +' -f wav -p -c 1 ' + audio.name + ' |\n')
					transcript = transcript_dir + "/"+ audio.stem + '.txt'

					trans = open(transcript).readlines()
					spk_list = []
					for line in trans:
						if line!='\n':
							if line[0] !='#':
								start, stop, spk = line.split(':')[0].split(' ')
								if spk=="A":
									spk = metafile[sess_id][1]
								else:
									spk = metafile[sess_id][3]
								spk_list.append([start, stop, spk])
								utt_id = spk+ '-' + audio.stem +  '_' + str(int(1000*float(start))) 	+ '-' + str(int(1000*float(stop)))
								segf.write(utt_id + ' ' + audio.stem + ' ' + start + ' '+ stop + '\n')
								uttf.write(utt_id + ' ' +spk +'\n')
	wavscpf.close()
	segf.close()
	uttf.close()
	return None


if __name__ == "__main__":

	parser = make_argument_parser()
	args = parser.parse_args()


	wavscp = open("./data/wav.scp", "w")
	segments = open("./data/segments", "w")
	utt2spk = open("./data/utt2spk", "w")
	print("output files created...")

	metaf = create_meta_dict(args.meta_file)
	print("metadata dict created")
	create_kaldi_files(audio_dir_root = args.audio_dir,
					   transcript_dir= args.transcript_dir,
					   metafile = metaf,
					   sph2pipe = args.sph2pipe,
					   uttf = utt2spk,
					   segf = segments,
					   wavscpf= wavscp)
	print("done!")
