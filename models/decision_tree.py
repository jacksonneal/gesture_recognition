import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from preprocessing.preprocessor import Preprocessor


class Node:
    """
    Node class of decision tree.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None,
                 value=None):
        """
        Initialize node with the given params.
        :param feature_index: index of feature being evaluated by node
        :param threshold: cutoff for delivering instances to left vs right child
        :param left: left child
        :param right: right child
        :param info_gain: information gain based on split of node
        :param value: value of leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        """
        Initialize the root of the decision tree to None and initialize the
        stopping conditions
        """
        # Start code here
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # End code here

    def build_tree(self, dataset, cur_depth=0):
        """
        This will be a recursive function to build the decision tree.
        dataset: The data that you will be using for your assignment
        cur_depth: Current depth of the tree
        Returns the leaf node.
        Steps:
          1. Separate the features and targets.
          2. Find the best split for features
          3. If stopping conditions are not met then recursively form the left
          and right subtree if info_gain is positive else return leaf value
        """
        # Start code here
        leaf_value = None

        # 1.
        targets = dataset[:, -1]
        features = dataset[:, :-1][0]
        # 2.
        best_split = self.get_best_split(dataset, len(dataset), len(features))
        if len(best_split["dataset_left"]) == 0 or len(best_split["dataset_right"]) == 0:
            best_split = None
        # 3.a
        if cur_depth < self.max_depth and len(
                dataset) > self.min_samples_split and best_split is not None:
            left = self.build_tree(best_split["dataset_left"], cur_depth + 1)
            right = self.build_tree(best_split["dataset_right"], cur_depth + 1)
            return Node(best_split["feature_index"], best_split["threshold"], left, right,
                        best_split["info_gain"])
        # 3.b
        else:
            leaf_value = self.calculate_leaf_value(targets)
        # End code here

        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        """
        Function to find out the best split by looping over each feature. For
            each feature loop over all feature values and calculate best_split
            based on the info_gain after spliting on that value
        dataset: input data
        num_samples: Number of samples present in the dataset
        num_features: Number of features in the dataset
        Returns the best split
        """

        # dictionary to store the best split and its details
        best_split = {}
        max_info_gain = -float("inf")

        # Start code here
        # loop over all the features in the data
        for feature_index in range(num_features):
            for sample_index in range(num_samples):
                threshold = dataset[sample_index][feature_index]
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                info_gain = self.information_gain(dataset[:, -1], dataset_left[:, -1],
                                                  dataset_right[:, -1])
                if info_gain > max_info_gain:
                    best_split["dataset_right"] = dataset_right
                    best_split["dataset_left"] = dataset_left
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["info_gain"] = info_gain
                    max_info_gain = info_gain
        # End code here

        return best_split

    def split(self, dataset, feature_index, threshold):
        """
        Function to split the data to the left child and right child in the decision tree
        dataset: input data
        feature_index: feature index used to locate the index of the feature in a particular row
        in the dataset
        threshold: threshold value based on which the split will be calculated
        Returns the left and right datavalues from the dataset
        """
        dataset_left = None
        dataset_right = None
        # Start code here
        # Hint: Use list comprehension to distinguish which values would be present in left and
        # right
        # subtree on the basis of threshold
        dataset_left, dataset_right = self.split_helper(dataset,
                                                        dataset[:, feature_index] <= threshold)
        # End code here

        return dataset_left, dataset_right

    def split_helper(self, arr, cond):
        """
        Function to split array based on boolean conditional array.
        arr: to split
        cond: boolean array indicating how to split
        Returns array of 2 arrays, first corresponding to positive cond, second to negative
        """
        return [arr[cond], arr[~cond]]

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """
        Function to calculate information gain. This function subtracts the combined information
        of the child node from the parent node.
        parent: value of parent node
        l_child: value of left child node
        r_child: value of right child node
        mode: based on which information gain will be calculated either entropy/gini index
        Returns the information gain
        Steps:
            1. Calculate relative sizes of child nodes w.r.t parent node
            2. Calculate gain on the with respect to the information gain parameter which will
            either be
            gini_index or entropy
        """
        # Start code here
        if mode == "gini":
            gain = self.gini_index(parent) - self.gini_index(l_child) - self.gini_index(r_child)
        else:
            gain = self.entropy(parent) - self.entropy(l_child) - self.entropy(r_child)
        # End code here

        return gain

    def entropy(self, y):
        """
        Function to calculate the entropy. Extracts the class labels and
        calculates the entropy.
        y: target labels
        Returns entropy
        """
        # Start code here
        vc = pd.Series(y).value_counts(normalize=True, sort=False)
        base = 2
        entropy = -(vc * np.log(vc) / np.log(base)).sum()
        # End code here

        return entropy

    def gini_index(self, y):
        """
        Function to calculate gini index. Extracts the class labels and
        calculates the gini index.
        y: target labels
        Returns gini index
        """
        # Start code here
        label_map = {}
        for label in y:
            if not label in label_map:
                label_map[label] = 0
            label_map[label] += 1
        sum = 0
        for value in label_map.values():
            sum += (value / len(y)) ** 2
        return 1 - sum
        # End code here

        return gini

    def calculate_leaf_value(self, Y):
        """
        Function to compute the value of leaf node i.e. returns the most occurring
        element in Y.
        Y: target labels
        Returns leaf node value
        """
        # Start code here
        return stats.mode(Y)[0][0]
        # End code here

    def print_tree(self, tree=None, indent=" "):
        """
        Function to print the tree. Use the pre-order traversal method to print the decision tree.
        # Do not make any changes in this function
        """

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X " + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end=" ")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end=" ")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        """
        Function to train the tree. Concatenate X, Y to
        create the dataset and call the build_tree function recursively
        X: Features
        Y: Target
        """
        # Start code here
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        # End code here

    def predict(self, X):
        """
        Prediction function to calculate the all the predictions of the matrix of features
        provided using make_predictions function
        X: Matrix of features
        Returns predictions using the make_prediction function
        """
        # Start code here
        predictions = [self.make_prediction(x, self.root) for x in X]
        # End code here

        return predictions

    def make_prediction(self, x, tree):
        """
        Function to predict a single datapoint. Recursively traverses the tree based on the
        threshold value until a leaf node is reached.
        x: data
        tree: current tree
        Returns predictions
        """
        # Start code here
        if tree.value is not None:
            return tree.value
        elif x[tree.feature_index] <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        # End code here


if __name__ == '__main__':
    print(sys.argv)
    X_train, y_train = Preprocessor.access_data_labels(sys.argv[1])
    X_test, y_test = Preprocessor.access_data_labels(sys.argv[2])

    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train, y_train)
    classifier.print_tree()
    y_pred = classifier.predict(X_test)

    print("Accuracy is: " + str(accuracy_score(y_test, y_pred)))
