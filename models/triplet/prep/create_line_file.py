from entrainment.config import *
metaf = open(fisher_meta, 'r')

reader = csv.reader(metaf)
metadata ={}
for row in reader:
	metadata[row[0]] = row[1:]

wavscpf = open('data/wav.scp', 'w')
segf = open('data/segments', 'w')
uttf = open('data/utt2spk', 'w')
linef = open('data/file2line', 'wb')
LineDict = {}
for dir in os.listdir(audio_dir_root):
	if "fisher_eng_tr_sp" in dir:
		# print(dir)
		subdir = audio_dir_root + "/" + dir + "/audio"
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
