import argparse
import matplotlib as plt

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

def plot_data(y, plot_save_path):
    x_points = [n for n, _ in enumerate(y)]
    print(len(x_points))
    plot = plt.scatter(x_points, y)
    plt.savefig(plot_save_path)
    plt.clf()
    print("plot saved to disk!")
    return plot



if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()



    for _, i in enumerate(args.data_list.strip().split(',')):
        print("plotting data for: ", i)
        y_data = open_file(args.data_dir + "/" + i)
        plot = plot_data(y_data, args.data_dir + "/" + i + ".png")
        print("number of data points: ", len(y_data))