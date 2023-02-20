import os
import csv
import pandas as pd
dir = "/Users/meghavarshinikrishnaswamy/transcripts"
unique_trials = ["T000719", "T000720"]
def open_files(dir):
    file_list = []
    for file in os.listdir(dir):
        ext = file.split('.')
        if ext[-1] == "txt":
            file_list.append(file)
    # print(file_list)
    return file_list

def extract_utterances(dir, unique_trials, file_ls):
    for i in unique_trials:
        print("starting with trial: ", i)
        filesavename = None
        i_df = None
        for file in file_ls:
            # pandas_frame = []
            filename = file.split(".")[0]
            data = filename.split("_")
            trial = data[2].split("-")
            team = data[3].split("-")
            speaker = data[4].split("-")
            if trial[1] == i:
                print("trial information found, processing transcript: ", filename)
                df = pd.read_csv(dir+"/"+file, sep='\t', encoding='utf8')
                print("data subset: ")
                print("rows: ", df.axes[0])
                print("cols: ", df.axes[1])
                if i_df is None:
                    filesavename = dir + "/"+ data[0]+ "_"+ data[1]+ "_"+ data[2]+ "_"+ data[3] + \
                                   "_"+ "Member-NA"+ "_"+ data[5] + "_"+ data[6] \
                                   + "_"+ data[7] + ".csv"
                    i_df = df
                else:
                    i_df = pd.concat([i_df, df], axis = 0)
            else:
                print("moving to next file")
        print(i_df.columns)
        #sort column by start time
        i_df.sort_values(by = ["start"])
        #remove empty columns
        i_df.dropna(how="all", axis=1, inplace=True)
        #remove colon in addressee columns
        i_df["addressee"] = i_df["addressee"].apply(lambda x: x.split(":")[0])
        # save csv file
        i_df.to_csv(filesavename, index = False, sep = "\t")
        print("trial dataframe:")
        print("rows: ", i_df.axes[0])
        print("cols: ", i_df.axes[1])
    return None

files = open_files(dir)
x = extract_utterances(dir, unique_trials, files)
