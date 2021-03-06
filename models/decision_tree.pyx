#!python
#cython: language_level=3
import copy
import os
import pandas as pd
import numpy as np
import json
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from models.model import Algo, Model
from collections import Counter

MAX_PARALLEL = 20


class Node(ABC):
    """
    Base node class.
    """

    @abstractmethod
    def debug_print(self, indent=" "):
        """
        Print node and children if any.
        :return: None
        """
        pass

    @abstractmethod
    def make_prediction(self, x):
        """
        Function to predict a single datapoint. Recursively traverses the tree based on the
        threshold value until a leaf node is reached.
        :param x: data
        :return: prediction
        """
        pass


class DecisionNode(Node):
    """
    Decision node of tree.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None):
        """
        Initialize node with the given params.
        :param feature_index: index of feature being evaluated by node
        :param threshold: cutoff for delivering instances to left vs right child
        :param left: left child
        :param right: right child
        :param info_gain: information gain based on split of node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

    def debug_print(self, indent=" "):
        print(f"FI: {self.feature_index}, Theta: {self.threshold}, IG: {self.info_gain}")
        print("%sleft:" % indent, end=" ")
        self.left.debug_print(indent + indent)
        print("%sright:" % indent, end=" ")
        self.right.debug_print(indent + indent)

    def make_prediction(self, x):
        # print(x.describe())
        if x.values[self.feature_index] <= self.threshold:
            return self.left.make_prediction(x)
        else:
            return self.right.make_prediction(x)


class LeafNode(Node):
    """
    Leaf node of tree.
    """

    def __init__(self, value=None):
        """
        Initialize the leaf node with the given value
        :param value:
        """
        self.value = value

    def debug_print(self, indent=" "):
        print(self.value)

    def make_prediction(self, x):
        return self.value


class SplitEngine:
    """
    Functional evaluation of a single potential split on the dataset.  Can be submitted to
    multiprocessing pool.
    """

    def __init__(self, parent_counts, mode):
        """
        Configure the dataset that we will evaluate splits for.
        :param parent_counts:
        :param mode:
        """
        self.parent_counts = parent_counts
        self.mode = mode

    def __call__(self, params):
        """
        Evaluate split and return stats.
        :param params: defines split
        :return: split, new datasets, info_gain
        """
        l_counts, r_counts, feature_index, threshold = params
        # threshold = self.dataset[sample_index][feature_index]
        # dataset_left, dataset_right = DecisionTreeClassifier.split(self.dataset, feature_index,
        #                                                            threshold)
        info_gain = DecisionTreeClassifier.information_gain(self.parent_counts,
                                                            l_counts,
                                                            r_counts,
                                                            self.mode)
        split = {"feature_index": feature_index, "threshold": threshold, "info_gain": info_gain}
        return split


class DecisionTreeClassifier(Algo):
    """
    Binary Decision Tree.
    """

    def debug_print(self):
        self.root.debug_print()

    def save(self, dest):
        obj = {
            "type": Model.decision_tree.value,
            "root": self.root,
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "max_split_eval": self.max_split_eval,
            "mode": self.mode,
            "num_valid_features": self.num_valid_features
        }
        with open(dest, "w") as f:
            json.dump(obj, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def load(src):
        with open(src, "r") as f:
            return json.load(f, object_hook=DecisionTreeClassifier.as_payload)

    @staticmethod
    def as_payload(dct):
        if "value" in dct:
            return LeafNode(dct["value"])
        elif "feature_index" in dct:
            return DecisionNode(dct["feature_index"], dct["threshold"],
                                dct["left"], dct["right"],
                                dct["info_gain"])
        else:
            return DecisionTreeClassifier(dct["min_samples_split"], dct["max_depth"],
                                          dct["max_split_eval"], dct["root"], dct["mode"] == "gini",
                                          dct["num_valid_features"])

    def __init__(self, min_samples_split=2, max_depth=2, max_split_eval=1000, root=None,
                 use_gini=False, num_valid_features=None):
        """
        Initialize the root of the decision tree to None and initialize the
        stopping conditions.
        :param min_samples_split: min samples in a decision node
        :param max_depth: max vertical depth of decision tree
        :param max_split_eval: max comparisons when determining split vals
        :param root: root node
        :param use_gini: whether to use gini index instead of entropy
        :param num_valid_features: if present, limit to number of randomly chosen features
        available for splitting when training
        """
        self.root = root
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_split_eval = max_split_eval
        self.pool = ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, os.cpu_count() - 1))
        self.mode = "gini" if use_gini else "entropy"
        self.num_valid_features = num_valid_features
        self.valid_feature_indices = None

    def __del__(self):
        """
        Destructor close thread pool.
        :return: None
        """
        self.pool.shutdown()

    def build_tree(self, dataset, cur_depth=0):
        """
        Recursive function to build decision tree.
        :param dataset: to split
        :param cur_depth: current depth
        :return: root node
        """
        """
        Steps:
          1. Separate the features and targets.
          2. Find the best split for features
          3. If stopping conditions are not met then recursively form the left
          and right subtree if info_gain is positive else return leaf value
        """
        print(f"cur_depth: {cur_depth}, num_samples: {dataset.shape[0]})")

        # 1.
        # targets = dataset.iloc[:, -1]
        # features = dataset.iloc[:, :-1][0]

        # 1.b establish valid feature indices
        if self.valid_feature_indices is None:
            if self.num_valid_features is None:
                self.valid_feature_indices = range(dataset.shape[1] - 1)
            else:
                self.valid_feature_indices = random.sample(range(dataset.shape[1] - 1),
                                                           self.num_valid_features)

        # 2.
        best_split = self.get_best_split(dataset, dataset.shape[0])
        if best_split is not None and (len(best_split["dataset_left"]) == 0 or len(
                best_split["dataset_right"]) == 0):
            best_split = None

        # 3.a
        if cur_depth < self.max_depth and dataset.shape[0] > self.min_samples_split \
                and best_split is not None:
            left = self.build_tree(best_split["dataset_left"], cur_depth + 1)
            right = self.build_tree(best_split["dataset_right"], cur_depth + 1)
            return DecisionNode(best_split["feature_index"], best_split["threshold"], left, right,
                                best_split["info_gain"])
        # 3.b
        else:
            return LeafNode(DecisionTreeClassifier.calculate_leaf_value(dataset.iloc[:, -1]))

    def get_best_split(self, dataset, num_samples):
        """
        Function to find out the best split by looping over each feature. For
            each feature loop over all feature values and calculate best_split
            based on the info_gain after splitting on that value.  Performs split evaluations in
            parallel.
        :param dataset: input data
        :param num_samples: num samples in dataset
        :return: best split
        """
        labels = dataset['Y']
        split_params = []
        for feature_index in self.valid_feature_indices:
            ds_sorted = dataset.sort_values(by=dataset.columns[feature_index])
            thresholds = ds_sorted.iloc[:, feature_index]
            cur_labels = ds_sorted['Y']
            means = self.moving_avg(thresholds.unique(), 2)
            l_counts = {}
            r_counts = Counter(labels)

            # print(thresholds.shape)
            # print(thresholds.describe())

            i = 1
            for j in range(len(means)):
                while i < thresholds.shape[0] and thresholds.iloc[i - 1] <= means[j]:
                    klass = cur_labels.iloc[i - 1]
                    if klass not in l_counts.keys():
                        l_counts[klass] = 0
                    l_counts[klass] += 1
                    r_counts[klass] -= 1

                    if thresholds.iloc[i - 1] == thresholds.iloc[i]:
                        i += 1
                        continue

                    split_params.append(
                        (copy.deepcopy(l_counts), copy.deepcopy(r_counts), feature_index, means[j]))
                    i += 1

        # if len(split_params) > self.max_split_eval * 2:
        #     split_params = random.sample(split_params, self.max_split_eval)

        split_engine = SplitEngine(Counter(labels), self.mode)
        splits = self.pool.map(split_engine, split_params)

        # splits = list(map(split_engine, split_params))

        best_split = max(splits, key=lambda x: x["info_gain"], default=None)
        if best_split is None:
            return best_split
        left, right = self.split(dataset, best_split["feature_index"], best_split["threshold"])
        best_split["dataset_left"] = left
        best_split["dataset_right"] = right
        return best_split

    @staticmethod
    def moving_avg(x, window):
        return np.convolve(x, np.ones(window), 'valid')

    @staticmethod
    def split(dataset, feature_index, threshold):
        """
        Function to split the data to the left child and right child in the decision tree.
        :param dataset: input data
        :param feature_index: feature to split on
        :param threshold: threshold to split feature on
        :return: left and right splits of dataset
        """
        return DecisionTreeClassifier.split_by_cond(dataset,
                                                    dataset.iloc[:, feature_index] <= threshold)

    @staticmethod
    def split_by_cond(arr, cond):
        """
        Function to split array based on boolean conditional array.
        arr: to split
        cond: boolean array indicating how to split
        Returns array of 2 arrays, first corresponding to positive cond, second to negative
        """
        return [arr[cond], arr[~cond]]

    @staticmethod
    def information_gain(parent, l_child, r_child, mode="entropy"):
        """
        Function to calculate information gain. This function subtracts the combined information
        of the child node from the parent node.
        :param parent: parent counts
        :param l_child: left child counts
        :param r_child: right child counts
        :param mode: type of information gain to be used
        :return: information gain
        """
        """
        Steps:
            1. Calculate relative sizes of child nodes w.r.t parent node
            2. Calculate gain on the with respect to the information gain parameter which will
            either be gini_index or entropy
        """
        if mode == "gini":
            gain = DecisionTreeClassifier.gini_index(parent) - DecisionTreeClassifier.gini_index(
                l_child) - DecisionTreeClassifier.gini_index(r_child)
        else:
            gain = DecisionTreeClassifier.entropy(parent) - DecisionTreeClassifier.entropy(
                l_child) - DecisionTreeClassifier.entropy(r_child)

        return gain

    @staticmethod
    def entropy(y):
        """
        Extracts the class labels and calculates the entropy.
        :param y: labels counts
        :return: entropy
        """
        vc = pd.Series(y.values())  #  pd.Series(y).value_counts(normalize=True, sort=False)
        base = 2
        entropy = -(vc * np.log(vc) / np.log(base)).sum()

        return entropy

    @staticmethod
    def gini_index(y):
        """
        Extracts the class labels and calculates the gini index.
        :param y: target labels counts
        :return: gini index
        """
        total = 0
        n = 0
        for value in y.values():
            n += value
        for value in y.values():
            total += (value / n) ** 2
        return 1 - total

    @staticmethod
    def calculate_leaf_value(Y):
        """
        Function to compute the value of leaf node i.e. returns the most occurring
        element in Y.
        :param Y: target labels
        :return: leaf node value
        """
        return stats.mode(Y)[0][0]

    def fit(self, X, Y):
        """
        Function to train the tree. Concatenate X, Y to
        create the dataset and call the build_tree function recursively
        :param X: Features
        :param Y: Target
        :return: None
        """
        X.insert(X.shape[1], "Y", Y)
        # dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(X)

    def predict(self, X):
        """
        Prediction function to calculate the all the predictions of the matrix of features
        provided using make_predictions function.
        :param X: Matrix of features
        :return: Predictions using the make_prediction function
        """
        return [self.root.make_prediction(x) for _, x in X.iterrows()]
