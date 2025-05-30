import h5py
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

#-------------------------------------------------
# Define the dataset class
#-------------------------------------------------
class EntDataset(Dataset):
	def __init__(self, file_name):
		self.file_name = file_name
		
	def __getitem__(self, idx):
		hf = h5py.File(self.file_name, 'r')
		X = hf['dataset'][idx,0:228]
		Y = hf['dataset'][idx,228:456]
		hf.close()
		return (X, Y)

	def __len__(self):
		hf = h5py.File(self.file_name, 'r')
		length=hf['dataset'].shape[0]
		hf.close()
		return length
#-------------------------------------------------
# Define the model
#-------------------------------------------------
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		# the model is run over
		zdim=30

		self.fc1 = nn.Linear(228, 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, zdim)
		self.bn2 = nn.BatchNorm1d(zdim)

		self.fc3 = nn.Linear(zdim, 128)
		self.bn3 = nn.BatchNorm1d(128)
		self.fc4 = nn.Linear(128, 228)
		self.bn4 = nn.BatchNorm1d(228)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		h1 = self.fc1(x)
		h1 = self.bn1(h1)
		h1= self.relu(h1)

		h2 = self.fc2(h1)
		z = self.bn2(h2)  # can experiment with this

		return z

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decode(self, z):
		h3 = self.fc3(z)
		h3 = self.bn3(h3)
		h3 = self.relu(h3)
		h4 = self.fc4(h3)
		# h4 = self.sigmoid(h4)
		h4 = self.bn4(h4)
		return h4

	def forward(self, x):
		# view reshapes the vector, with given dimensions
		z = self.encode(x.view(-1, 228)) 
		
		# tested this, not used
		# z = self.reparameterize(mu, logvar)
		return self.decode(z)

	def embedding(self, x):
		z = self.encode(x.view(-1, 228))
		return z

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x1, x2):
	BCE = F.smooth_l1_loss(recon_x1, x2.view(-1, 228), size_average=False)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114

	return BCE

def lp_distance(x1, x2, p):
	dist = torch.dist(x1, x2,p)
	return dist
