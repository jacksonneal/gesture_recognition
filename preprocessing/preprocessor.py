import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocessor:
    """
    Preprocessor used for data preparation before any training or analysis.
    """

    @staticmethod
    def join_datasets(src, dest):
        """
        Join the dataset csv's found in the given src directory and place the resulting csv at
        the destination path.
        :param src: path to directory of csv's to join
        :param dest: path to file destination to write joined result
        :return: None
        """
        datasets = glob.glob(os.path.join(src, "*.csv"))
        joined = pd.concat((pd.read_csv(f, header=None) for f in datasets), ignore_index=True)
        joined.to_csv(dest, header=False, index=False)

    @staticmethod
    def train_test_split(src, dest, test_pct):
        """
        Split the dataset src file given into the given percentage of test and train examples,
        placing the result in the
        given destination directory.
        :param src: dataset file to split
        :param dest: directory to put splits
        :param test_pct: percentage of examples to be test
        :return: None
        """
        dataframe = pd.read_csv(src, header=None)
        dataframe = dataframe.sample(frac=1)  # Shuffle
        data = np.asarray(dataframe.iloc[:, :-1])
        labels = np.asarray(dataframe.iloc[:, -1])
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=float(test_pct),
                                                            random_state=42)
        train = np.column_stack((x_train, y_train))
        test = np.column_stack((x_test, y_test))
        pd.DataFrame(train).to_csv(os.path.join(dest, "train.csv"), header=False, index=False)
        pd.DataFrame(test).to_csv(os.path.join(dest, "test.csv"), header=False, index=False)

    @staticmethod
    def access_data_labels(file):
        """
        Access the separate data and labels of the given file.
        :param file: to read
        :return: data, labels
        """
        dataframe = pd.read_csv(file, header=None)
        return dataframe.iloc[:, :-1].values, dataframe.iloc[:, -1].values.reshape(-1, 1)


if __name__ == "__main__":
    if sys.argv[1] == "join":
        Preprocessor.join_datasets(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "train_test_split":
        Preprocessor.train_test_split(sys.argv[2], sys.argv[3], sys.argv[4])
