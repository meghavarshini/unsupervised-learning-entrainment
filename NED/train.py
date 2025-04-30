#To Run, use: CUDA_VISIBLE_DEVICES=1 python train.py --no-cuda
from ecdc import *
import csv
from datetime import datetime

#------------------------------------------------------------------
#Uncomment for parsing inputs
def make_argument_parser():
	parser = argparse.ArgumentParser(description='VAE MNIST Example')
	parser.add_argument('--model_name', type=str,
						default= "./baseline_1_models/trained_NED_l1.pt",
						help="name of model file")
	parser.add_argument('--h5_directory', type=str,
						default="./baseline_1_h5",
						help='location of h5 files')
	parser.add_argument('--batch-size', type=int, default=128, metavar='N',
						help='input batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='enables CUDA training'),
	parser.add_argument("--cuda_device", default=1, help="set device"),
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	
	return parser

def compute_y_stats_from_file(file_path, featDim:int = 228):
    with h5py.File(file_path, 'r') as hf:
        y_all = hf['dataset'][:, featDim:2*featDim]  # N x featDim
        y_tensor = torch.tensor(y_all)  # convert to tensor all at once
    y_mean = y_tensor.mean(dim=0)
    y_std = y_tensor.std(dim=0)
    return y_mean, y_std

def model_setup(model_name, seed: int, cuda: bool, cuda_device: int, data_directory):
	print(model_name)

	if os.path.exists(model_name):
		print("model file available for update: ", model_name)
	else:
		print("model file not found")

	torch.manual_seed(seed)
	if cuda:
		torch.cuda.set_device(cuda_device)
		torch.cuda.manual_seed(seed)

	# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	# f = h5py.File('interaction_Fisherfeats.h5','r')
	# dataset=f['dataset']

	y_mean, y_std = compute_y_stats_from_file(data_directory + "/" + "train_Fisher_nonorm.h5")
	print(f"Mean, SD calculated for norming")

	# Now, create train and validation datasets using y_mean and y_std
	fdset_train = EntDataset(data_directory + "/" + "train_Fisher_nonorm.h5", y_mean, y_std)
	train_loader = torch.utils.data.DataLoader(fdset_train, batch_size=32, shuffle=True)

	fdset_val = EntDataset(data_directory + "/" + "val_Fisher_nonorm.h5", y_mean, y_std)
	val_loader = torch.utils.data.DataLoader(fdset_val, batch_size=128, shuffle=True)

	model = VAE().double()
	if cuda:
		model.cuda(cuda_device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	return model, optimizer, train_loader, val_loader, y_mean, y_std

def train(each_epoch, model, train_loader, optimizer, cuda, cuda_device: int):
	model.train()
	train_loss = 0
	count = 0
	for batch_idx, (data, y_data) in enumerate(train_loader):
		data = Variable(data)
		y_data = Variable(y_data)
		if cuda:
			data = data.cuda(cuda_device)
			y_data = y_data.cuda(cuda_device)

		optimizer.zero_grad()

		recon_batch = model(data)
		loss = loss_function(recon_batch, y_data)
		loss.backward()

		if each_epoch == 1 and batch_idx == 0:  # Just sample from first batch of first epoch
			print("Sample targets:", recon_batch[:5])       # First 5 targets
			print("Sample predictions:", y_data[:5])   # Corresponding model outputs		
			
			# Optional: Check ranges
			print("Targets — min:", y_data.min().item(), 
					"max:", y_data.max().item(), 
					"mean:", y_data.mean().item())

			print("Outputs — min:", recon_batch.min().item(), 
					"max:", recon_batch.max().item(), 
					"mean:", recon_batch.mean().item())
		
		train_loss += loss.data
		print(f"Run: {count} loss: {loss.data}")
		count += 1
		optimizer.step()
		# if batch_idx % args.log_interval == 0:
			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				# epoch, batch_idx * len(data), len(train_loader.dataset),
				# 100. * batch_idx / len(train_loader),
				# loss.data[0] / len(data)))
	train_loss /=  len(train_loader.dataset)
	print(f"====> Epoch: {each_epoch} Average loss: {train_loss:.4f}")

	return train_loss

#Lines 88,89 depreciated
# https://stackoverflow.com/questions/61720460/volatile-was-removed-and-now-had-no-effect-use-with-torch-no-grad-instread
def validate(model, val_loader, cuda, cuda_device: int, y_mean, y_std):
	model.eval()
	
	val_loss = 0
	count = 0

	for i, (data, y_data) in enumerate(val_loader):
		if cuda:
			data = data.cuda(cuda_device)
			y_data = y_data.cuda(cuda_device)
			# Move y_mean and y_std to same device as model outputs
			y_mean = y_mean.to(cuda_device)
			y_std = y_std.to(cuda_device)
		# data = Variable(data, volatile=True)
		# y_data = Variable(y_data, volatile=True)
		data = Variable(data)
		y_data = Variable(y_data)
		
		# forward pass
		recon_batch = model(data)

		## use the following if using any metric that needs original scale.
		# # De-normalize the output
		# outputs_denormalized = recon_batch * y_std + y_mean
		# # De-normalize the targets (Y)
		# y_data_denormalized = y_data * y_std + y_mean

		# val_loss += loss_function(recon_batch, y_data).data[0]
		val_loss += loss_function(recon_batch, y_data).data
		
		print(f"====> Validation Loss for Run {count}: {val_loss:.4f}")
		count+=1
		

	val_loss /= len(val_loader.dataset)
	print(f"====> Average Validation Set Loss: {val_loss:.4f}")
	return val_loss


if __name__ == "__main__":
	parser = make_argument_parser()
	args = parser.parse_args()
	cuda_availability = not args.no_cuda and torch.cuda.is_available()
	torch.cuda.set_device(args.cuda_device)
	print("cuda availability: ", cuda_availability)
	print("gpu details: ", torch.cuda.get_device_name(args.cuda_device))

	# Check if Model exists:
	if os.path.isfile(args.model_name):
			print(f"The model '{args.model_name}' exists. Training will rewrite it.")
	else:
		print(f"The model '{args.model_name}' does not exist. Creating file...")

	# Check if model directory exists:
	model_directory_path = os.path.dirname(args.model_name)
	
	# Check if H5 directory exists
	if os.path.isdir(model_directory_path):
		print(f"The directory for storing the trained model '{model_directory_path}' exists. Continue...")
	else:
		print(f"The directory for storing the trained model '{model_directory_path}' does not exist. Creating it...")
		os.makedirs(model_directory_path, exist_ok=True)
		print("Rechecking for model: ", os.path.isdir(model_directory_path))
	
	# Check if H5 directory exists
	if os.path.isdir(args.h5_directory):
		print(f"The directory with training files '{args.h5_directory}' exists. Continue...")
	else:
		print(f"The directory with training files '{args.h5_directory}' does not exist.")
		sys.exit(1)

	Tloss =[]
	Vloss =[]
	save_loss_data =[]
	best_loss=np.inf
	print("This is Sparta!!")
	
	baseline_model, baseline_optimizer, baseline_train_loader, baseline_val_loader, y_mean, y_std = \
		model_setup(model_name = args.model_name, seed=args.seed, cuda=cuda_availability, 
					cuda_device = args.cuda_device, data_directory=args.h5_directory)
	print("Initial Model Setup Complete")

# for epoch in range(1, 3):
#Notes- torch.save saves both the state dict as well as the optimizer-
# if we have a model and all we want to do is use it, then we save the state dict. But if we want
# further train, fine-tune, then we need both he optimizer as well as the state dict.
	# Save loss
	loss_file_name = model_directory_path+"/" + datetime.now().strftime("%Y%m%d-%H%M")+"_model-loss.csv"
	with open(loss_file_name, mode='w', newline='') as file:
		writer = csv.writer(file)		
		# Write header
		writer.writerow(('Epoch', 'Train Loss', 'Val Loss'))

		# Loop for training through epochs and saving data
		for epoch in range(1, args.epochs + 1):
			tloss = train(each_epoch = epoch, model = baseline_model,
							train_loader= baseline_train_loader,
							optimizer= baseline_optimizer, cuda = cuda_availability, 
							cuda_device = args.cuda_device)
			vloss = validate(model= baseline_model,
							val_loader= baseline_val_loader,
							cuda= cuda_availability, cuda_device = args.cuda_device,
							y_mean=y_mean, y_std=y_std)
			Tloss.append(tloss)
			Vloss.append(vloss)
			writer.writerow((epoch, tloss.item(), vloss.item()))
			save_loss_data.append((epoch, tloss.item(), vloss.item()))
			
			if vloss < best_loss:
				best_loss = vloss
				best_epoch = epoch
				print("Epoch: ", epoch, "Validation Loss: ", vloss.item())
				torch.save(baseline_model, args.model_name)

print(save_loss_data)
	
