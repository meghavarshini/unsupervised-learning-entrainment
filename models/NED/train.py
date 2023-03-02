# from aeent import *
from ecdc import *
from test import test, load_h5
from matplotlib import pyplot as plt

#------------------------------------------------------------------
#Uncomment for parsing inputs

parser = argparse.ArgumentParser(description='NED training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
	help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
	help='number of epochs to train (default: 10)')  # increased default from 10 to 100
parser.add_argument('--no-cuda', action='store_true', default=False,
	help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
	help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
	help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("argparse loaded")


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# f = h5py.File('interaction_Fisherfeats.h5','r')
# dataset=f['dataset']


dataset_id = 'Fisher_acoustic'  # applies for ip and op
norm_id = 'nonorm'  # applies for ip and op
dim_id = '30dim'     # applies for op only   # CHECK ecdc.py zdim value
loss_id = 'l1'
method_id = ''


fdset = EntDataset('/home/tomcat/entrainment/NED_files/mini/train_' + dataset_id + '_' + norm_id +'.h5')

train_loader = torch.utils.data.DataLoader(fdset, batch_size=128, shuffle=True)


fdset_val = EntDataset('/home/tomcat/entrainment/NED_files/mini/val_' + dataset_id + '_' + norm_id + '.h5')

val_loader = torch.utils.data.DataLoader(fdset_val, batch_size=128, shuffle=True)

print("data loaded")

model = VAE().double()
if args.cuda:
    model.cuda()
#Model LR hasn't converged to a best loss in 100 epochs- so we make the LR biggers to make observations about convergence
#We run the risk of the step size being too big
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    print("training...")
    model.train()
    train_loss = 0
    for batch_idx, (data, y_data) in enumerate(train_loader):
        data = Variable(data)
#        print("data dimensions: ",data.size())
        y_data = Variable(y_data)
 #       print("data dimensions: ", y_data.size())
        #exit()

        if args.cuda:
            data = data.cuda()
            y_data = y_data.cuda()

        optimizer.zero_grad()

        # recon_batch = model(data)
        # VAE's forward function contains both encoder and decoder.
        # But for NED, only encoded items used for caculating distances
        recon_batch = model(data)
        loss = loss_function(recon_batch, y_data)
        loss.backward()
        print("train loss calcutation: ", batch_idx % args.log_interval)
        #IndexError: invalid index of a 0-dim tensor.
        # Use `tensor.item()` in Python to convert a 0-dim tensor to a number
        # todo: testing out changing .data for .item on loss calculations
        # train_loss += loss.data
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                # loss.data / len(data))))
                loss.item() / len(data))))
    train_loss /=  len(train_loader.dataset)
    print(('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss)))

    return train_loss


def validate(epoch):
    model.eval()
    val_loss = 0
    for i, (data, y_data) in enumerate(val_loader):
        if args.cuda:
            data = data.cuda()
            y_data = y_data.cuda()
        # Used `with torch.no_grad():` instead of Volatile
        #Tells the (y) variable (torch tensor) that it won't be updated with
        # variable info
        with torch.no_grad():
            data = Variable(data)
            y_data = Variable(y_data)
        recon_batch = model(data)
        # recon_batch = model.encode(data)
        # encoded_y = model.encode(y_data) # so that y is swapped with the encoded
        val_loss += loss_function(recon_batch, y_data).data

    val_loss /= len(val_loader.dataset)
    print(('====> Validation set loss: {:.4f}'.format(val_loss)))
    return val_loss


Tloss =[]
Vloss =[]
dev2loss = []
fake_dev2loss = []
dev2result = []
best_loss=np.inf
print("This is Sparta!!")

for epoch in range(1, args.epochs + 1):
    #  print("test")
    tloss = train(epoch)
    vloss = validate(epoch)
    Tloss.append(tloss)
    Vloss.append(vloss)

    # need test data
    # todo: add second dev partition and add its path in here
    dev2_path = "/home/tomcat/entrainment/NED_files/mini/dev2_Fisher_acoustic_nonorm.h5"
    dev2_data = load_h5(dev2_path)

    # test out this version of the model on testing task
    # use a separate DEV partition
    # todo: add flexibility to have p=2 if needed
    dev2_loss, fake_dev2_loss, dev2_result = test(X_test = dev2_data, model = model, cuda = args.cuda, p=1)
    # append these to the holders
    dev2loss.append(dev2_loss)
    fake_dev2loss.append(fake_dev2_loss)
    dev2result.append(dev2_result)

    # patience for increases/plateau in loss
    # number of epochs to run without loss decreasing
    #   beyond best loss
    patience = 2

    # print("vloss")
    if vloss < best_loss:
        patience = 2
        print("model updated")
        best_loss = vloss
        best_epoch = epoch
        # pdb.set_trace()
        torch.save(model, '/home/tomcat/entrainment/NED_files/mini/models/trained_' + dataset_id + '_' + norm_id + '_'+ loss_id + '_'+ dim_id + '.pt')
    else:
        patience -= 1
        if patience == 0:
            break

print("plotting training loss values...")
plt.scatter(epoch_no, Tloss)
plt.scatter(epoch_no, Vloss)
plt.savefig("/home/tomcat/entrainment/NED_files/mini/loss_train_plot.png")
plt.clf()

print("plotting training loss values...")
plt.scatter(epoch_no, dev2loss)
plt.savefig("/home/tomcat/entrainment/NED_files/mini/loss_dev2_plot.png")
plt.clf()


print("plotting dev3 results...")
plt.scatter(epoch_no, dev2result)
plt.savefig("/home/tomcat/entrainment/NED_files/mini/dev2_result_plot.png")
