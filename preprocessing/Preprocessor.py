import os
import pandas as pd
import glob


class Preprocessor:
    """
    Preprocessor used for data preparation before any training or analysis.
    """

    @staticmethod
    def join_datasets(src, dest):
        """
        Join the dataset csv's found in the given src directory and place the resulting csv at the destination path.
        :param src: path to directory of csv's to join
        :param dest: path to file destination to write joined result
        :return: None
        """
        datasets = glob.glob(os.path.join(src, "*.csv"))
        joined = pd.concat((pd.read_csv(f, header=None) for f in datasets), ignore_index=True)
        joined.to_csv(dest)
