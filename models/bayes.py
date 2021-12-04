import random

import numpy as np
import json

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

from models.model import Algo, Model
from preprocessing.preprocessor import Preprocessor


class NaiveBayesClassifier(Algo):

    def __init__(self, num_valid_features=None):
        self.num_valid_features = num_valid_features

    def debug_print(self):
        pass

    @staticmethod
    def compute_mean(nums):
        """
        Function to compute the mean of a list of values
        nums: list of values
        Returns the mean of the values
        :rtype: object
        """
        return np.average(nums)

    @staticmethod
    def compute_standard_deviation(nums):
        """
        Function to compute the standard deviation of a list of values
        nums: list of values
        Returns the standard deviation of the values
        """
        return np.std(nums)

    @staticmethod
    def splitup_dataset(dataset, class_column):
        """
        Function to split the dataset according to the class
        dataset: the data you will be using for the assignment
        class_column: the index of the column that contains class values
        Returns a dictionary having mapping for each class value to the list of
        records it is present in.
        """
        keys = np.unique(np.array(dataset)[:, class_column])
        split_data = {}
        for value in keys:
            split_data[value] = list()

        for i in range(len(dataset)):
            split_data[dataset[i][class_column]].append(dataset[i][:class_column])

        return split_data

    def calculate_stats(self, dataset, valid_columns):
        """
        This function will calculate statistics i.e. mean, standard deviation
        and number of records for each column in the given data
        dataset: data for which statistics needs to be computed
        Returns a list containing the calculated statistics for each column
        NOTE: Returned list should have a statistics as a tuple 
            i.e. (mean, std_dev, count) for each column
        """
        dataset = np.array(dataset)
        stats = list()
        count = np.shape(dataset)[0]

        for column_index in valid_columns:
            column_data = dataset[:, column_index]
            mean = self.compute_mean(column_data)
            std_dev = self.compute_standard_deviation(column_data)
            stats.append((column_index, mean, std_dev, count))
        return stats

    def calculate_stats_by_class(self, dataset):
        """
        This function should be used to calculate the statistics for each class
        dataset: data for which statistics needs to be computed
        Returns a dictionary containing statistics for each class.
        NOTE: Use splitup_dataset and calculate_stats in this method and 
        last column contains the class values
        """
        self.model = {}
        self.training_data_len = len(dataset)

        self.valid_columns = range(len(dataset[0]) - 1)
        # Ensemble chooses from limited number of features
        if self.num_valid_features is not None:
            self.valid_columns = random.sample(self.valid_columns, self.num_valid_features)

        split_data = self.splitup_dataset(dataset, len(dataset[0]) - 1)
        for class_value in split_data:
            self.model[class_value] = self.calculate_stats(split_data[class_value],
                                                           self.valid_columns)

    @staticmethod
    def probability(x, mean, std_dev):
        """
        Helper for prediction
        mean: the calculated mean
        std_dev: the calculated std_dev
        Returns the probability of x in the class for the given mean and standard deviation
        """
        z = (x - mean) / std_dev
        return 1 / std_dev / np.sqrt(2 * np.pi) * np.exp(-0.5 * z ** 2)

    def compute_probabilities_by_class(self, record, class_value):
        """
        Given the statistics for each class, this function computes
        the probability of given record belonging to that class for all classes.
        record: record whose probability has to be found.
        NOTE: Use the precomputed class wise probability stores in self.model and 
        the probability function
        """
        probability = self.model[class_value][0][2] / self.training_data_len
        for i in range(len(self.model[class_value])):
            col, mean, std_dev, count = self.model[class_value][i]
            probability *= self.probability(record[col], mean, std_dev)

        return probability

    def make_prediction(self, record):
        """
        This function predicts the class to which the given record
        belongs.
        record: record to classify
        Returns the class to which the record belongs.
        """
        best_option = None
        best_probability = 0
        for class_value in self.model:
            prob = self.compute_probabilities_by_class(record, class_value)
            if prob > best_probability:
                best_probability = prob
                best_option = class_value

        return best_option

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.calculate_stats_by_class(dataset)

    def predict(self, test):
        predictions = list()
        for row in test:
            output = self.make_prediction(row)
            predictions.append(output)
        return predictions

    def save(self, dest):
        obj = self.model
        obj["type"] = Model.bayes.value
        with open(dest, "w") as f:
            json.dump(obj, f, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    @staticmethod
    def load(src):
        loaded_classifier = NaiveBayesClassifier()
        loaded_classifier.training_data_len = 0
        loaded_classifier.model = {}

        with open(src, "r") as f:
            obj = json.load(f)

        for key in obj:
            if key == "type":
                continue
            loaded_classifier.model[float(key)] = obj[key]
            loaded_classifier.training_data_len += obj[key][0][2]

        return loaded_classifier


if __name__ == '__main__':
    X_train, y_train = Preprocessor.access_data_labels("..\\datasets\\train_test_split\\train.csv")
    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    classifier.save("data.txt")

    print("from Scratch")
    classifier = NaiveBayesClassifier.load("data.txt")
    X_test, y_test = Preprocessor.access_data_labels("..\\datasets\\train_test_split\\test.csv")
    predictions = classifier.predict(X_test)
    #accuracy(y_test, predictions)
    print(confusion_matrix(y_test, predictions))
    print("Accuracy is: " + str(accuracy_score(y_test, predictions)))

    print("\nSciKit")  # This comes out with identical results
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    predictions = gnb.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
