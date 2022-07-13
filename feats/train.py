from ecdc import *
#------------------------------------------------------------------

print(model_name)
if os.path.exists(model_name):
    print("model file available for update: ", model_name)
else:
    print("model file not found")
#Uncomment for parsing inputs
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--data_directory', type=str, default="/home/tomcat/entrainment/feat_files/data",
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# f = h5py.File('interaction_Fisherfeats.h5','r')
# dataset=f['dataset']


fdset = EntDataset(args.data_directory + "/" + "train_Fisher_nonorm.h5")

train_loader = torch.utils.data.DataLoader(fdset, batch_size=128, shuffle=True)


fdset_val = EntDataset(args.data_directory + "/" + "val_Fisher_nonorm.h5")

val_loader = torch.utils.data.DataLoader(fdset_val, batch_size=128, shuffle=True)

model = VAE().double()
if args.cuda:
	model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y_data) in enumerate(train_loader):
        data = Variable(data)
        y_data = Variable(y_data)

        if args.cuda:
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
          epoch, train_loss)))

    return train_loss

#Lines 88,89 depreciated
# https://stackoverflow.com/questions/61720460/volatile-was-removed-and-now-had-no-effect-use-with-torch-no-grad-instread
def validate(epoch):
    model.eval()
    val_loss = 0
    for i, (data, y_data) in enumerate(val_loader):
        if args.cuda:
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


Tloss =[]
Vloss =[]
best_loss=np.inf
print("This is Sparta!!")

# for epoch in range(1, 3):
for epoch in range(1, args.epochs + 1):
    tloss = train(epoch)
    vloss = validate(epoch)
    Tloss.append(tloss)
    Vloss.append(vloss)
    if vloss < best_loss:
        best_loss = vloss
        best_epoch = epoch
        print("epoch: ", vloss, "epoch: ", epoch)
        torch.save(model, model_name)
