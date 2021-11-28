from enum import Enum
from abc import ABC, abstractmethod


class Model(Enum):
    decision_tree = "decision-tree"
    bagging = "bagging"

    def __str__(self):
        return self.value


class Algo(ABC):

    @abstractmethod
    def print(self):
        """
        Print for debug.
        :return: None
        """
        raise NotImplemented

    @abstractmethod
    def save(self, dest):
        """
        Serialize and save to file dest.
        :param dest: file
        :return: None
        """
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def load(src):
        """
        Deserialize the given file and return the model.
        :param src: to load
        :return: algo
        """
        raise NotImplemented

    @abstractmethod
    def fit(self, X, Y):
        """
        Fit dataset to train model
        :param X:
        :param Y:
        :return:
        """
        raise NotImplemented

    @abstractmethod
    def predict(self, X):
        """
        Prediction function to calculate the all the predictions of the matrix of features
        provided.
        :param X: Matrix of features
        :return: Predictions
        """
        raise NotImplemented
