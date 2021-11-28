import argparse

from models.bagging import Bagging
from preprocessing.preprocessor import Preprocessor
from models.decision_tree import DecisionTreeClassifier
from enum import Enum
from models.model import Model
from sklearn.metrics import accuracy_score


class Action(Enum):
    train = "train"
    test = "test"

    def __str__(self):
        return self.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Act on gesture recognition dataset.")

    # First positional argument must indicate model type
    parser.add_argument("model", type=Model, choices=list(Model), help="Model to use.")

    # Second positional argument must indicate action on model
    parser.add_argument("action", type=str, nargs=3,
                        help="Indicate action and relevant files: "
                             "(train, training csv, destination for trained model)"
                             "(test, model file to load, test csv)")

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

    # Bagging Argument: Number of bootstrapped samples per model
    parser.add_argument("--k", type=int, default=1000, help="Num bootstrapped samples per model.")

    # Bagging Argument: Number of DecisionTrees to run
    parser.add_argument("--n-dt", type=int, default=10, dest="num_decision_trees",
                        help="Number of decision trees in ensemble.")

    opts = parser.parse_args()

    model = None
    if opts.action[0] == "train":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier(opts.min_split, opts.max_depth, opts.max_split_eval,
                                           None, opts.gini)
        elif opts.model == Model.bagging:
            trees = [DecisionTreeClassifier(opts.min_split, opts.max_depth, opts.max_split_eval,
                                            None, opts.gini) for _ in
                     range(opts.num_decision_trees)]
            model = Bagging(trees, opts.k)
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        X_train, y_train = Preprocessor.access_data_labels(opts.action[1])
        model.fit(X_train, y_train)
        model.save(opts.action[2])
        if opts.print:
            model.print()

    elif opts.action[0] == "test":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier.load(opts.action[1])
        elif opts.model == Model.bagging:
            model = Bagging.load(opts.action[1])
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        X_test, y_test = Preprocessor.access_data_labels(opts.action[2])
        if opts.print:
            model.print()
        predictions = model.predict(X_test)
        print("Accuracy is: " + str(accuracy_score(y_test, predictions)))

    else:
        raise ValueError(f"Unsupported action type {opts.action}")
