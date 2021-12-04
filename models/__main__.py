import argparse
import os
from random import randrange

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import decision_tree
from models.ensemble import Ensemble
from models.bayes import NaiveBayesClassifier
from preprocessing.preprocessor import Preprocessor
from decision_tree import DecisionTreeClassifier
# from models.decision_tree import DecisionTreeClassifier
from enum import Enum
from models.model import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

NUM_FOLDS = 5


class Action(Enum):
    train = "train"
    test = "test"

    def __str__(self):
        return self.value


def custom_style(row):
    color = 'white'
    if row.values[-1] != row.values[-2]:
        color = 'red'
    return ['background-color: %s' % color] * len(row.values)


def cross_validation_split(ds, n_folds):
    dataset_split = list()
    dataset_copy = ds.values.tolist()
    fold_size = int(len(ds) / n_folds)
    for _ in range(n_folds):
        cur_fold = list()
        while len(cur_fold) < fold_size:
            index = randrange(len(dataset_copy))
            cur_fold.append(dataset_copy.pop(index))
        dataset_split.append(cur_fold)
    return dataset_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Act on gesture recognition dataset.")

    # First positional argument must indicate model type
    parser.add_argument("model", type=Model, choices=list(Model), help="Model to use.")

    # Second positional argument must indicate action on model
    parser.add_argument("action", type=str, nargs=4,
                        help="Indicate action and relevant files: "
                             "({train|test}, train.csv, test.csv, model.json)")

    # Optional log results argument can be used with test action
    parser.add_argument("--save", type=str, default=None,
                        help="Save test results to the given directory.")

    # Optional print model
    parser.add_argument("--print", action="store_true", help="Debug print.")

    # DecisionTree Argument: Min instances in node
    parser.add_argument("--min-split", type=int, default=2, dest="min_split",
                        help="Min instances in node of tree.")

    # DecisionTree Argument: Max depth of tree
    parser.add_argument("--max-depth", type=int, default=3, dest="max_depth",
                        help="Max depth of decision tree.")

    # DecisionTree Argument: Max split evaluations
    parser.add_argument("--max-split-eval", type=int, default=1000, dest="max_split_eval",
                        help="Max split evaluations when determining node config.")

    # DecisionTree Argument: GiniIndex
    parser.add_argument("--gini", action="store_true", help="Use gini index instead of entropy.")

    # Ensemble Argument: Number of samples per model
    parser.add_argument("--k", type=int, default=1000, help="Num bootstrapped samples per model.")

    # Ensemble Argument: Number of DecisionTrees to run
    parser.add_argument("--n-dt", type=int, default=10, dest="num_decision_trees",
                        help="Number of decision trees in ensemble.")

    # Ensemble Argument: Number of Bayes to run
    parser.add_argument("--n-nb", type=int, default=10, dest="num_naive_bayes",
                        help="Number of naive bayes classifiers in ensemble.")

    # Ensemble Argument: Number of features to be considered in each model
    parser.add_argument("--nf", type=int, default=None, dest="num_valid_features",
                        help="Number of features to be sampled and considered in each model of "
                             "ensemble.")

    # Ensemble Argument: boost by testing against the given test dataset
    parser.add_argument("--boost", action="store_true",
                        help="Indicate that we should do boosting.")

    # Configurable parallelism
    parser.add_argument("--parallel", type=int, default=20,
                        help="Max number of parallel processes to use.")

    # Configure cross validation when testing
    parser.add_argument("--cv", action="store_true", dest="cross_validate",
                        help="Indicate to perform K-Fold Cross Validation in addition to test "
                             "action.")

    opts = parser.parse_args()

    decision_tree.MAX_PARALLEL = opts.parallel

    model = None
    if opts.action[0] == "train":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier(opts.min_split, opts.max_depth, opts.max_split_eval,
                                           None, opts.gini, opts.num_valid_features)

        elif opts.model == Model.bayes:
            model = NaiveBayesClassifier()

        elif opts.model == Model.ensemble:
            bayes = [NaiveBayesClassifier(opts.num_valid_features) for _ in
                     range(opts.num_naive_bayes)]

            trees = [DecisionTreeClassifier(opts.min_split, opts.max_depth, opts.max_split_eval,
                                            None, opts.gini, opts.num_valid_features) for _ in
                     range(opts.num_decision_trees)]

            algos = []
            for i in range(len(bayes)):
                algos.append(bayes[i])
                if i < len(trees):
                    algos.append(trees[i])
            for i in range(len(bayes), len(trees)):
                algos.append(trees[i])

            model = Ensemble(algos, opts.k, opts.boost)
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        X_train, y_train = Preprocessor.access_data_labels(opts.action[1])
        s = StandardScaler()
        X_train = pd.DataFrame(s.fit_transform(X_train), columns=X_train.columns)
        model.fit(X_train, y_train)
        model.save(opts.action[3])
        if opts.print:
            model.debug_print()

    elif opts.action[0] == "test":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier.load(opts.action[3])
        elif opts.model == Model.bayes:
            model = NaiveBayesClassifier.load(opts.action[3])
        elif opts.model == Model.ensemble:
            model = Ensemble.load(opts.action[3])
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        s = StandardScaler()
        X_train, y_train = Preprocessor.access_data_labels(opts.action[1])
        s.fit_transform(X_train)

        if opts.cross_validate:
            # cross_val_score(model, X, y, cv=NUM_FOLDS)
            dataset = pd.read_csv(opts.action[2], header=None)
            folds = cross_validation_split(dataset, NUM_FOLDS)
            scores = []
            for fold in folds:
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()
                for record in fold:
                    record_copy = list(record)
                    test_set.append(record_copy)
                    record_copy[-1] = None
                test_x = [record[:-1] for record in fold]
                test_x = pd.DataFrame(s.transform(test_x))
                test_y = pd.DataFrame([record[-1] for record in fold])
                predicted = model.predict(test_x)
                accuracy = accuracy_score(test_y, predicted)
                scores.append(accuracy)
            print(f"{NUM_FOLDS}-Fold Cross Validation: {scores}")
            print(f"Average Score: {sum(scores) / len(scores)}")

        X_test, y_test = Preprocessor.access_data_labels(opts.action[2])
        X_test = pd.DataFrame(s.transform(X_test), columns=X_test.columns)

        if opts.print:
            model.print()
        predictions = model.predict(X_test)
        print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
        print(classification_report(y_test, predictions))
        if opts.save is not None:
            # Save all results, highlight incorrect predictions
            res = np.append(X_test.values, y_test.values.reshape(-1, 1), axis=1)
            res = np.insert(res, res.shape[1], predictions, axis=1)
            res_df = pd.DataFrame(res)

            df_styled = res_df.style.apply(custom_style, axis=1)
            with open(os.path.join(opts.save, "x_y_pred.html"), "w") as f:
                f.write(df_styled.render())

            # Save confusion matrix
            cm = confusion_matrix(y_test, predictions)
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(os.path.join(opts.save, "cm.csv"))

            # Save arguments and accuracy of test run
            with open(os.path.join(opts.save, "log.txt"), "w") as f:
                f.write("Args:" + str(opts) + "\n")
                f.write("Accuracy: " + str(accuracy_score(y_test, predictions)))

            # Plot CM
            f, ax = plt.subplots(figsize=(15, 15))
            sns.set(font_scale=1.4)
            sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Purples", linecolor="black",
                        ax=ax)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix Validation set")
            plt.show()
    else:
        raise ValueError(f"Unsupported action type {opts.action}")
