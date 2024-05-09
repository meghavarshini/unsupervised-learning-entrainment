#To Run, use: CUDA_VISIBLE_DEVICES=1 python train.py --no-cuda
from ecdc import *
#------------------------------------------------------------------
#Uncomment for parsing inputs
def make_argument_parser():
	parser = argparse.ArgumentParser(description='VAE MNIST Example')
	parser.add_argument('--model_name', type=str,
						default= "/home/tomcat/entrainment/feat_files/baseline_1_models/trained_VAE_nonorm_nopre_l1.pt",
						help="name of model file")
	parser.add_argument('--h5_directory', type=str,
						default="/home/tomcat/entrainment/feat_files/baseline_1_h5",
						help='location of h5 files')
	parser.add_argument('--batch-size', type=int, default=128, metavar='N',
						help='input batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	# args = parser.parse_args()
	# args.cuda = not args.no_cuda and torch.cuda.is_available()
	return parser

def model_setup(model_name, seed, cuda, data_directory):
	print(model_name)

	if os.path.exists(model_name):
		print("model file available for update: ", model_name)
	else:
		print("model file not found")

	torch.manual_seed(seed)
	if cuda:
		torch.cuda.manual_seed(seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# f = h5py.File('interaction_Fisherfeats.h5','r')
# dataset=f['dataset']


	fdset = EntDataset(data_directory + "/" + "train_Fisher_nonorm.h5")
	train_loader = torch.utils.data.DataLoader(fdset, batch_size=128, shuffle=True)


	fdset_val = EntDataset(data_directory + "/" + "val_Fisher_nonorm.h5")
	val_loader = torch.utils.data.DataLoader(fdset_val, batch_size=128, shuffle=True)

	model = VAE().double()
	if cuda:
		model.cuda()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	return model, optimizer, train_loader, val_loader

def train(each_epoch, model, train_loader, optimizer, cuda):
	model.train()
	train_loss = 0
	for batch_idx, (data, y_data) in enumerate(train_loader):
		data = Variable(data)
		y_data = Variable(y_data)

		if cuda:
			data = data.cuda()
			y_data = y_data.cuda()

		optimizer.zero_grad()

		recon_batch = model(data)
		loss = loss_function(recon_batch, y_data)
		loss.backward()
		print("loss data: ",loss.data)
		train_loss += loss.data
		# train_loss += loss.data[0]
		optimizer.step()
		# if batch_idx % args.log_interval == 0:
			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				# epoch, batch_idx * len(data), len(train_loader.dataset),
				# 100. * batch_idx / len(train_loader),
				# loss.data[0] / len(data)))
	train_loss /=  len(train_loader.dataset)
	print(('====> Epoch: {} Average loss: {:.4f}'.format(
		  each_epoch, train_loss)))

	return train_loss

#Lines 88,89 depreciated
# https://stackoverflow.com/questions/61720460/volatile-was-removed-and-now-had-no-effect-use-with-torch-no-grad-instread
def validate(model, val_loader, cuda):
	model.eval()
	val_loss = 0
	for i, (data, y_data) in enumerate(val_loader):
		if cuda:
			data = data.cuda()
			y_data = y_data.cuda()
		# data = Variable(data, volatile=True)
		# y_data = Variable(y_data, volatile=True)
		data = Variable(data)
		y_data = Variable(y_data)
		recon_batch = model(data)
		val_loss += loss_function(recon_batch, y_data).data
		# val_loss += loss_function(recon_batch, y_data).data[0]

	val_loss /= len(val_loader.dataset)
	print(('====> Validation set loss: {:.4f}'.format(val_loss)))
	return val_loss


if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	print("cuda availability: ", args.cuda)
	print("gpu details: ", torch.cuda.get_device_name(0))


	Tloss =[]
	Vloss =[]
	best_loss=np.inf
	print("This is Sparta!!")
	baseline_model, baseline_optimizer, baseline_train_loader, baseline_val_loader = \
		model_setup(args.model_name, args.seed, args.cuda, args.h5_directory)

# for epoch in range(1, 3):
#Notes- torch.save saves both the state dict as well as the optimizer-
# if we have a model and all we want to do is use it, then we save the state dict. But if we want
# further train, fine-tune, then we need both he optimizer as well as the state dict.
	for epoch in range(1, args.epochs + 1):
		tloss = train(each_epoch = epoch, model = baseline_model,
					  train_loader= baseline_train_loader,
					  optimizer= baseline_optimizer, cuda = args.cuda)
		vloss = validate(model= baseline_model,
						 val_loader= baseline_val_loader,
						 cuda= args.cuda)
		Tloss.append(tloss)
		Vloss.append(vloss)
		if vloss < best_loss:
			best_loss = vloss
			best_epoch = epoch
			print("epoch: ", vloss, "epoch: ", epoch)
			torch.save(baseline_model, args.model_name)
			#model = torch.load(PATH)
			#model.eval()
			#parameters- check they are non-empty