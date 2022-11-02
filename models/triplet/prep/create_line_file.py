import pickle
import argparse

def make_argument_parser():
	parser = argparse.ArgumentParser(
        				description="Processing filepaths and values required for setup")
	parser.add_argument("fisher_corpus",
						default = "/media/mule/projects/ldc/fisher-corpus",
						description = "corpus directory")
	parser.add_argument("kaldi_output",
						default="/home/tomcat/entrainment/feat_files/kaldi_output",
						description= "directory for kaldi_output from create_kaldi.py")
	return parser

# metaf = open(args.fisher_corpus + "/Fisher_meta.csv", 'r')
def line_file_creator(fisher_corpus, wavscpf1, segf1, uttf1, linef1, audio_dir_root1):
	## read csv file with speaker details:
	reader = csv.reader(metaf)
	metadata ={}
	for row in reader:
		metadata[row[0]] = row[1:]

	## read kaldi output files:
	wavscpf = open(wavscpf1, 'w')
	segf = open(segf, 'w')
	uttf = open(uttf1, 'w')
	linef = open(linef1, 'wb')

	LineDict = {}
	for dir in os.listdir(audio_dir_root1):
		if "fisher_eng_tr_sp" in dir:
			# print(dir)
			subdir = audio_dir_root1 + "/" + dir + "/audio"
			# print(subdir)
			for subsubdir in os.listdir(subdir):
				if "0" in subsubdir:
					# print(subsubdir)
					for audio in os.listdir(subdir + '/' + subsubdir):
						audio_path = subdir + '/' + subsubdir + '/'+ audio
						# print("audio_path", audio_path)
						audio = audio.split(".")[0]
						sess_id = audio.split('_')[-1]
						wavscpf.write(audio + ' '+ sph2pipe+' -f wav -p -c 1 ' + audio_path + ' |\n')
						transcript = transcript_dir + "/" + audio + '.txt'
						trans = open(transcript).readlines()
						spk_list = []
						j = 0
						for i, line in enumerate(trans):
							if line!='\n':
								if line[0] !='#':
									start, stop, spk = line.split(':')[0].split(' ')
									if spk=="A":
										spk = metadata[sess_id][1]
									else:
										spk = metadata[sess_id][3]
									spk_list.append([start, stop, spk])
									utt_id = spk + '-' + audio + '_' + str(int(1000*float(start))) + '-' + str(int(1000*float(stop)))
									LineDict[utt_id] = j
									j +=1
									segf.write(utt_id + ' ' + audio + ' ' + start + ' '+ stop + '\n')
									uttf.write(utt_id + ' ' +spk +'\n')
					# print(audio, spk_list[0][2])

	pickle.dump(LineDict, linef)

	wavscpf.close()
	segf.close()
	uttf.close()

	return None

if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()

	wavscpf = args.kaldi_output + "/wav.scp"
	segf = args.kaldi_output + "/segments"
	uttf = args.kaldi_output + "/utt2spk"
	linef = args.kaldi_output + "/file2line"
	audio_dir_root = args.fisher_corpus + "/fisher_eng_tr_sp_LDC2004S13_zip"

	line = line_file_creator(fisher_corpus = args.fisher_corpus, wavscpf1 = wavscpf,
							 segf1 = segf, utt1 = utt, linef1 = linef,
							 audio_dir_root1 = audio_dir_root)




