from sklearn.metrics import cohen_kappa_score
import pandas as pd


class IRR:
    def __init__(self, dataframe1_paths, dataframe2_paths):
        """
        Initialize an IRR object
        :param dataframe1_paths: A list of filepaths for annotator 1
        :param dataframe2_paths: A list of filepaths for annotator 2
        """
        self.df1 = self._concat_files(dataframe1_paths)
        self.df2 = self._concat_files(dataframe2_paths)

        self.valid_args = {"addressee": ['all', 'engineer', 'medic', 'transporter']}

    def get_data_distribution(self, annotation_type):
        """
        Get the distribution of data across classes for a given annotation type
        :param annotation_type: a string with the name of the annotation type
            'sentiment' for sentiment annotation
            'emotion' for emotion annotation
        """
        return self.df1[annotation_type].value_counts(), self.df2[annotation_type].value_counts()

    def get_kappas(self, annotation_type):
        """
        Get cohen's kappa scores for sentiment and emotion annotations
        :param annotation_type: a string with the name of the annotation type
            'sentiment' for sentiment annotation
            'emotion' for emotion annotation
        :returns the cohen's kappa scores and value counts for the annotation
            type in each df
        """
        if annotation_type in self.valid_args.keys():
            valid = self.valid_args[annotation_type]
            df1 = self.df1[self.df1[annotation_type].isin(valid)]
            df2 = self.df2[self.df2[annotation_type].isin(valid)]
        else:
            df1 = self.df1
            df2 = self.df2
        kappa = cohen_kappa_score(df1[annotation_type], df2[annotation_type])

        return kappa

    def _concat_files(self, list_of_files):
        """
        Concatenate a list of files
        :return:
        """
        all_files = None
        for file in list_of_files:
            pd_df = pd.read_csv(file)
            if all_files is None:
                all_files = pd_df
            else:
                all_files = pd.concat([all_files, pd_df], axis=0)

if __name__ == "__main__":
