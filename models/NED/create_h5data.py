from entrainment.config import *

SEED=448
frac_train = 0.8
frac_val = 0.1


def clean_feat(XX, dim):
	ind = []
	for i, pair in enumerate(XX):
		x = pair[0:dim]
		y = pair[dim:]
		if np.any(x) and np.any(y) and (not np.any(np.isnan(x))) and (not np.any(np.isnan(y))):
			ind.append(i)
	XX = XX[ind,:]
	return XX

# Create h5 files

dataset_id = 'Fisher_acoustic'
norm_id = 'nonorm'
dim = 228
# Making files for the test partition, with train and val commented out. Try uncommenting, and make this happen
data_dir = feats_dir

# Try to generate this sessList correctly
sess_files = os.path.isfile("data/sessList.txt")
if sess_files == 1:
	with open("data/sessList.txt", 'r') as f:
		temp = f.read().splitlines()
		if len(temp) == len(sorted(glob.glob(feats_dir + '/*.csv'))):
			print("list of shuffled files exists, importing...")
			sessList = temp
else:
	sessList= sorted(glob.glob(feats_dir + '/*.csv'))
	print("creating a list of shuffled feature files and saving to disk...")
	# print("sessList", sessList)
	random.seed(SEED)
	random.shuffle(sessList)
	with open("data/sessList.txt", 'w') as f:
		f.writelines( "%s\n" % i for i in sessList)

	with open("data/sessList.txt", 'r') as f:
		sessList = f.read().splitlines()

## Alternative to 27-45
# sessList = sorted(glob.glob(data_dir + '/*.csv'))
## sessList= [f for f in sorted(glob.glob(data_dir + '/*.csv')) if int(os.path.basename(f).split('.')[0].split('_')[-2]) < 800]
# random.seed(SEED)
# random.shuffle(sessList)

num_files_all = len(sessList)
num_files_train = int(np.ceil((frac_train*num_files_all)))
num_files_val = int(np.ceil((frac_val*num_files_all)))
num_files_test = num_files_all - num_files_train - num_files_val
print("num_files_all: ", num_files_all)
print("num_files_train: ", num_files_train)
sessTrain = sessList[:num_files_train]
sessVal = sessList[num_files_train:num_files_val+num_files_train]
sessTest = sessList[num_files_val+num_files_train:]
print("Total File = ", len(sessTrain) + len(sessVal) + len(sessTest))

############ Uncomment the following chunks one by one to create data files ###################

###### Create Train Data file ######
temp_trainfile = os.getcwd()+"/data/tmp.csv"
try:
    os.remove(temp_trainfile)
except OSError:
    pass
ftmp = open(temp_trainfile, 'a')
for sess_file in sorted(sessTrain):
	start = time.time()
	print("sess_file: ", sess_file)
	########## ToDo: find out IPU feat files that don't have enough rows ########
	xx = np.genfromtxt(sess_file, delimiter= ",")
	xx = np.hstack((xx[0:-1,:], xx[1:,:]))
	xx = clean_feat(xx, dim)
	nn = xx.shape[0]
	np.savetxt(ftmp, xx, delimiter=',')
	print ('Train: ' +  sess_file + '  '+"{0:.2f}".format(time.time() - start) + '  '+ str(nn))

ftmp.close()
start = time.time()
X_train = np.genfromtxt(temp_trainfile, delimiter= ",")
X_train = X_train.astype('float64')
# os.remove(temp_trainfile)

print ('Reading Train takes  '+"{0:.2f}".format(time.time() - start) )

start = time.time()
hf = h5py.File('data/train_' + dataset_id + '_' + norm_id + '.h5', 'w')
hf.create_dataset('dataset', data=X_train)
hf.close()
print ('Writing Train takes '+"{0:.2f}".format(time.time() - start) )



###### Create Val Data file ######

# X_val = np.empty(shape=(0, 0), dtype='float64' )
# temp_valfile = os.getcwd()+"/data/tmp.csv"
# ftmp = open(temp_valfile, 'a')
# for sess_file in sorted(sessVal):
# 	start = time.time()
# 	print(sess_file)
# 	xx = np.genfromtxt(sess_file, delimiter= ",")
# 	xx = np.hstack((xx[0:-1,:], xx[1:,:]))
# 	xx = clean_feat(xx, dim)
# 	nn = xx.shape[0]
# 	np.savetxt(ftmp, xx, delimiter=',')
# 	print ('Val: ' +  sess_file + '  '+"{0:.2f}".format(time.time() - start) + '  '+ str(nn))
#
# ftmp.close()
# start = time.time()
# X_val = np.genfromtxt(temp_valfile, delimiter= ",")
# X_val = X_val.astype('float64')
# os.remove(temp_valfile)
#
# print ('Reading Val takes  '+"{0:.2f}".format(time.time() - start) )
#
# start = time.time()
# hf = h5py.File('data/val_' + dataset_id + '_' + norm_id + '.h5', 'w')
# hf.create_dataset('dataset', data=X_val)
# hf.close()
# print ('Writing Val takes '+"{0:.2f}".format(time.time() - start) )


###### Create Test Data file ######
# temp_testfile = temp_testfile
# ftmp = open(temp_testfile, 'a')
#
# spk_base = 1
# for sess_file in sessTest:
# 	xx = np.genfromtxt(sess_file, delimiter= ",")
# 	xx = np.hstack((xx[0:-1,:], xx[1:,:]))
# 	xx = clean_feat(xx, dim)
# 	N = xx.shape[0]
# 	if np.mod(N,2)==0:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 	else:
# 		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
# 		spk_label = np.append(spk_label, spk_base)
# 	xx = np.hstack((xx, spk_label.T.reshape([N,1])))
# 	spk_base += 1
# 	np.savetxt(ftmp, xx, delimiter=',')
# 	print('Test: ' +  sess_file , xx.shape[1])
#
# 	if xx.shape[1]!=913:
# 		print(sess_file)
# ftmp.close()
# X_test = np.genfromtxt(temp_testfile, delimiter= ",")
# X_test = X_test.astype('float64')
# hf = h5py.File('data/test_' + dataset_id + '_' + norm_id + '.h5', 'w')
# hf.create_dataset('dataset', data=X_test)
# hf.close()


# os.remove(temp_testfile)
