import argparse
from matplotlib import pyplot as plt

def make_argument_parser():
    parser = argparse.ArgumentParser(description='NED training')
    parser.add_argument('--data_dir', type=str,
                        default= "/home/tomcat/entrainment/NED_files/mini/models",
                        help="name of directory where loss data is stored")
    parser.add_argument('--data_list', type=str,
                        default="dev2loss, tloss, vloss, dev2result",
                        help='list of files in the data directory that need to be plotted')
    return parser

def open_file(file_path):
    print(file_path)
    with open(file_path, "r") as f: lines = [float(line.strip("\n,[,]")) for line in f]
    print("number of data points: ", len(lines))
    return lines

def plot_data(y_dataset_list:list= ["tloss", "vloss"], plot_save_path,\
              plot_title:str = "Loss Plot"):

    x_points = [n for n, _ in enumerate(y[0])]
    print(len(x_points))
    for ls in y_dataset_list:
        if len(y_dataset_list) == len(x_points) and len(x_points) is not None:
            plot = plt.plot(x_points, ls)
        else:
            Raise Exception("your loss datasets do not have the same array size")

    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig(plot_save_path)
    plt.clf()
    print("plot saved to disk!")
    return plot



if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    loss_data_fname = [i for i in args.data_list.strip().split(',') if "loss" in i]
    loss_data = [ open_file(args.data_dir + "/" + i.strip()) for i in loss_data_fname.strip()]
    print(len(loss_data))

    # for _, i in enumerate(args.data_list.strip().split(',')):
    # for _, i in enumerate(loss_data):
    #     print("plotting data for: ", i)
    #     y_data = open_file(args.data_dir + "/" + i.strip())
    #     plot = plot_data(y_data, \
    #                      plot_save_path = args.data_dir + "/" + i.strip() + ".png",\
    #                      plot_title= "Loss Data by Epoch")
    #     print("number of data points: ", len(y_data))