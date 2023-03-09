import os
from ecdc import *
from test import test, load_h5
from matplotlib import pyplot as plt
from datetime import datetime

#------------------------------------------------------------------
#Uncomment for parsing inputs
def make_argument_parser():
    parser = argparse.ArgumentParser(description='NED training')
    parser.add_argument('--model_dir', type=str,
                        default= "/home/tomcat/entrainment/NED_files/mini/models",
                        help="name of directory where model is stored")
    parser.add_argument('--h5_directory', type=str,
                        default="/home/tomcat/entrainment/NED_files/mini",
                        help='location of h5 files')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    return parser

def model_setup(seed, cuda, data_directory, model_directory):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # f = h5py.File('interaction_Fisherfeats.h5','r')
    # dataset=f['dataset']

    dataset_id = 'Fisher_acoustic'  # applies for ip and op
    norm_id = 'nonorm'  # applies for ip and op
    dim_id = '30dim'     # applies for op only   # CHECK ecdc.py zdim value
    loss_id = 'l1'
    method_id = ''

    model_name = model_directory + '/trained_' \
                       + dataset_id + '_' + norm_id + '_'+ loss_id + '_'+ dim_id + '.pt'
    if os.path.exists(model_name):
        print("model file available for update: ", model_name)
    else:
        print("model file not found")

    fdset = EntDataset(data_directory + '/train_' + dataset_id + '_' + norm_id +'.h5')
    train_loader = torch.utils.data.DataLoader(fdset, batch_size=128, shuffle=True)

    fdset_val = EntDataset(data_directory + '/val_' + dataset_id + '_' + norm_id + '.h5')
    val_loader = torch.utils.data.DataLoader(fdset_val, batch_size=128, shuffle=True)

    print("data loaded")

    model = VAE().double()
    if cuda:
        model.cuda()
    #Model LR hasn't converged to a best loss in 100 epochs- so we make the LR biggers to make observations about convergence
    #We run the risk of the step size being too big
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, train_loader, val_loader, model_name


def train(each_epoch, model, train_loader, optimizer, cuda):
    print("training...")
    # set model to training model
    model.train()
    # set loss to 0
    train_loss = 0

    # for every batch in the dataloader
    for batch_idx, (data, y_data) in enumerate(train_loader):
        # convert xs and ys to torch tensors
        data = torch.tensor(data, requires_grad=True)
        y_data = torch.tensor(y_data, requires_grad=True)
        # data = Variable(data)
        # y_data = Variable(y_data)

        # move to cuda if needed
        if cuda:
            data = data.cuda()
            y_data = y_data.cuda()

        # zero gradient on optimizer
        optimizer.zero_grad()

        # VAE's forward function contains both encoder and decoder.
        # But for NED, only encoded items used for caculating distances
        # feed xs through model to get reconstructed vectors
        recon_batch = model(data)
        # calculate the loss
        loss = loss_function(recon_batch, y_data)
        # backprop
        loss.backward()

        print("train loss calculation: ", batch_idx % args.log_interval)

        # Use `tensor.item()` in Python to convert a 0-dim tensor to a number
        train_loss += loss.item()

        # update gradients
        optimizer.step()

        # if it's time to log the training status
        if batch_idx % args.log_interval == 0:
            print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data))))

    # divide loss by length of dataset
    train_loss /= len(train_loader.dataset)

    print(('====> Epoch: {} Average loss: {:.4f}'.format(
          each_epoch, train_loss)))

    return train_loss


def validate(model, val_loader, cuda):
    print("processing validation set...")
    time_before_setting_eval = datetime.now()
    model.eval()
    time_after_setting_eval = datetime.now()
    print(f"Setting model to eval took: {time_after_setting_eval - time_before_setting_eval}")

    val_loss = 0
    for i, (data, y_data) in enumerate(val_loader):
        # Used `with torch.no_grad():` instead of Volatile
        #Tells the (y) variable (torch tensor) that it won't be updated with
        with torch.no_grad():
            data = torch.tensor(data, requires_grad=False)
            y_data = torch.tensor(data, requires_grad=False)

        # move to cuda if needed
        if cuda:
            data = data.cuda()
            y_data = y_data.cuda()

        recon_batch = model(data)
        # encoded_y = model.encode(y_data) # so that y is swapped with the encoded
        val_loss += loss_function(recon_batch, y_data).item()

    val_loss /= len(val_loader.dataset)
    print(('====> Validation set loss: {:.4f}'.format(val_loss)))
    return val_loss


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    epoch_no = []
    Tloss =[]
    Vloss =[]
    dev2loss = []
    fake_dev2loss = []
    dev2result = []
    best_loss = np.inf
    print("This is Sparta!!")

    ned_model, ned_optimizer, ned_train_loader, ned_val_loader, model_name = \
        model_setup(model_directory= args.model_dir, seed= args.seed,
                    cuda= args.cuda, data_directory = args.h5_directory)
    print("model loaded")
    # Second Dev set for testing model at every epoch
    dev2_path = args.h5_directory + "/dev2_Fisher_acoustic_nonorm.h5"
    dev2_data = load_h5(dev2_path)

    open(args.model_dir + "/dev2loss", 'w').close()
    open(args.model_dir + "/tloss", 'w').close()
    open(args.model_dir + "/vloss", 'w').close()
    open(args.model_dir + "/dev2result", 'w').close()

    file_dev2loss = open(args.model_dir + "/dev2loss", 'a')
    file_tloss = open(args.model_dir + "/tloss", 'a')
    file_vloss = open(args.model_dir + "/vloss", 'a')
    file_dev2result = open(args.model_dir + "/dev2result", 'a')

    for epoch in range(1, args.epochs + 1):
        #  print("test")
        tloss = train(each_epoch = epoch, model = ned_model,
                      train_loader= ned_train_loader,
                      optimizer= ned_optimizer, cuda = args.cuda)
        vloss = validate(model= ned_model,
                         val_loader= ned_val_loader,
                         cuda= args.cuda)

        epoch_no.append(epoch)
        Tloss.append(tloss)
        file_tloss.write(str(tloss) + "\n")
        Vloss.append(vloss)
        file_vloss.write(str(vloss) + "\n")


        # todo: add flexibility to have p=2 if needed
        dev2_loss, fake_dev2_loss, dev2_result = \
            test(X_test = dev2_data, model = ned_model, cuda = args.cuda, p=1)
        # append these to the holders

        dev2loss.append(dev2_loss)
        file_dev2loss.write(str(dev2_loss)+"\n")
        fake_dev2loss.append(fake_dev2_loss)
        dev2result.append(dev2_result)
        file_dev2result.write(str(dev2_result) + "\n")

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
            torch.save(ned_model, model_name)
        else:
            patience -= 1
            if patience == 0:
                break

    file_tloss.close()
    file_vloss.close()
    file_dev2loss.close()
    file_dev2result.close()