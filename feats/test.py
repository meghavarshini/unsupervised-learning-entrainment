from ecdc import *
import argparse
import pdb
import numpy as np
import random
import os

os.getcwd()

SEED=448

def make_argument_parser():
    parser = argparse.ArgumentParser(description='entrainment testing')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--hff', default= '/home/tomcat/entrainment/feat_files/baseline_1_h5/test_Fisher_nonorm.h5',
                        help='location of h5 file with the test data')
    parser.add_argument('--model_name',
                        default= "/home/tomcat/entrainment/feat_files/baseline_1_models/trained_VAE_nonorm_nopre_l1.pt",
                        help='name associated with the trained model')
    # args = parser.parse_args()
    return parser



def load_h5(file):
    print("h5 file: ", file)
    hff = h5py.File(file, 'r')
    test = np.array(file['dataset'])
    return test

def model_testing(model_name, X_test,cuda):
    model = VAE().double()
    model = torch.load(model_name)
    model.eval()
    if cuda:
        model.cuda()
    if 'l1' in model_name:
        p=1
    elif 'l2' in model_name:
        p=2
    else:
        print("need better model name")
        p=2

        pdb
    test_loss = 0
    fake_test_loss = 0
    Loss=[]
    Fake_loss = []
    # for batch_idx, (x_data, y_data) in enumerate(test_loader):
    for idx, data in enumerate(X_test):
        x_data = data[:228]
        y_data = data[228:-1]
        idx_same_spk =list(np.where(X_test[:,-1]==data[-1]))[0]

        ll = random.choice(list(set(idx_same_spk) -set([idx])))
        spk = int(data[-1])

        x_data = Variable(torch.from_numpy(x_data))
        y_data = Variable(torch.from_numpy(y_data))

        y_fake_data = X_test[ll,228:-1]

        y_fake_data = Variable(torch.from_numpy(y_fake_data))

        if cuda:
            x_data = x_data.cuda()
            y_data = y_data.cuda()
            y_fake_data = y_fake_data.cuda()

        recon_batch = model(x_data)


        z_x = model.embedding(x_data)
        z_y = model.embedding(y_data)
        # z_x = x_data
        # z_y = y_data
        loss_real = lp_distance(z_x, z_y, p).data
        # loss_real = loss_function(z_x, z_y, mu, logvar)


        z_y_fake = model.embedding(y_fake_data)
        # z_y_fake = y_fake_data

        loss_fake = lp_distance(z_x, z_y_fake, p).data
        # loss_fake = loss_function(z_x, z_y_fake, mu, logvar)

        test_loss += loss_real
        fake_test_loss += loss_fake

    # this is inefficient- find a way to do everything on CPU
        Loss.append(loss_real.cpu())
        Fake_loss.append(loss_fake.cpu())

    # print loss_real, loss_fake
    test_loss /= X_test.shape[0]
    fake_test_loss /= X_test.shape[0]
    Loss = np.array(Loss)
    Fake_loss = np.array(Fake_loss)

    print("Total Real Loss: "+ str(test_loss))
    print("Total Fake Loss: " + str(fake_test_loss))

    print(float(np.sum(Loss < Fake_loss))/Loss.shape[0])
    return None


if __name__ == "__main__":
    os.getcwd()
    parser = make_argument_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    X_test = load_h5(args.hff)
    test_run = model_testing(args.model_name, X_test, args.cuda)
