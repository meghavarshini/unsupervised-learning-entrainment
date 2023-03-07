import argparse
import matplotlib as plt

def make_argument_parser():
    parser = argparse.ArgumentParser(description='NED training')
    parser.add_argument('--data_dir', type=str,
                        default= "/home/tomcat/entrainment/NED_files/mini/models",
                        help="name of directory where loss data is stored")
    parser.add_argument('--file_list', type=str,
                        default="[dev2loss, tloss, vloss, dev2result]",
                        help='list of files in the data directory that need to be plotted')
    return parser

def open_file(file_path):
    with open(file_path, "r") as f: lines = [int(line.strip()) for line in f]
    print("number of data points: ", len(data))
    return lines

def plot_data(y, plot_save_path):
    x_points = [i for i in range(1, len(y))]
    print(len(x_points))
    plot = plt.scatter(x, y)
    plt.savefig(plot_save_path)
    plt.clf()
    print("plot saved to disk!")
    return plot



if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    for i,n in enumerate(args.data_list):
        print("plotting data for: "+i)
        y_data = open_file(args.filepath + "/" + i)
        plot = plot_data(y_data,args.filepath + "/" + i+ ".png" )
        print("number of data points: ", len(y_data))