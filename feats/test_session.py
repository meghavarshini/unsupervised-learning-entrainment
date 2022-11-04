from ecdc import *
import argparse
import pdb
import numpy as np
import random
import os

os.getcwd()
SEED=448

#------------------------------------------------------------------
#Uncomment for parsing inputs
def make_argument_parser():
    parser = argparse.ArgumentParser(description='entrainment testing')
    parser.add_argument('--hff', action='store_true',
            default="/home/tomcat/entrainment/feat_files/baseline_1_h5/test_Fisher_nonorm.h5",
            help='enables CUDA training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--model_name',
            default= "/home/tomcat/entrainment/feat_files/baseline_1_models/trained_VAE_nonorm_nopre_l1.pt",
            help='name associated with the trained model')
    args = parser.parse_args()
    return parser



# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

def load_h5(file):
    print("h5 file: ", file)
    file = h5py.File(file, 'r')
    test = np.array(file['dataset'])
    return test
# hff = h5py.File('data/test_Fisher_nonorm.h5', 'r')
# # hff = h5py.File('data/test_Fisher_nonorm_nopre.h5', 'r')
# X_test = np.array(hff['dataset'])


def model_testing(model_name, X_test, cuda= False):
    ##NameError: name 'VAE' is not defined
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

    results = []
    for k in range(10):

        Loss=[]
        Fake_loss = []
        # for batch_idx, (x_data, y_data) in enumerate(test_loader):
        N = int(X_test[-1,-1])

        for spk_pair in range(1, N+1):

            idx_same_spk = list(np.where(X_test[:,-1]==spk_pair))[0]

            test_loss = 0
            fake_test_loss = 0

            for idx in idx_same_spk:

                ll = random.choice(list(set(idx_same_spk) - set([idx])))
                x_data = X_test[idx,:228]
                y_data = X_test[idx,228:-1]
                y_fake_data = X_test[ll,228:-1]

                x_data = torch.from_numpy(x_data)
                y_data = torch.from_numpy(y_data)
                y_fake_data = torch.from_numpy(y_fake_data)

                if cuda:
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()
                    y_fake_data = y_fake_data.cuda()

                z_x = model.embedding(x_data)
                z_y = model.embedding(y_data)
                z_y_fake = model.embedding(y_fake_data)

                loss_real = lp_distance(z_x, z_y, p).data[0]
                loss_fake = lp_distance(z_x, z_y_fake, p).data[0]
                test_loss += loss_real
                fake_test_loss += loss_fake
            # pdb.set_trace()
            Loss.append(test_loss)
            Fake_loss.append(fake_test_loss)
            # print loss_real, loss_fake

        Loss=np.array(Loss)
        Fake_loss=np.array(Fake_loss)

        total_test_loss = np.sum(Loss)/Loss.shape[0]
        total_fake_test_loss = np.sum(Fake_loss)/Loss.shape[0]

        print("Total Real Loss:"+str(total_test_loss) + "Total Fake Loss:" + str(total_fake_test_loss))
        print(float("mean difference in real vs fake loss: ", np.sum(Loss < Fake_loss))/Loss.shape[0])

        results.append(float(np.sum(Loss < Fake_loss))/Loss.shape[0])


    print("results: ", np.mean(np.array(results)))

    return None

if __name__ == "__main__":
    os.getcwd()
    parser = make_argument_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    X_test1 = load_h5(args.hff)
    test_run = model_testing(args.model_name, X_test1, args.cuda)