import argparse
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

    # DecisionTree Argument: Min instances in node
    parser.add_argument("--min-split", type=int, default=2, dest="min_split",
                        help="Min instances in node of tree.")

    # DecisionTree Argument: Max depth of tree
    parser.add_argument("--max-depth", type=int, default=3, dest="max_depth",
                        help="Max depth of decision tree.")

    opts = parser.parse_args()

    model = None
    if opts.action[0] == "train":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier(opts.min_split, opts.max_depth)
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        X_train, y_train = Preprocessor.access_data_labels(opts.action[1])
        model.fit(X_train, y_train)
        model.save(opts.action[2])

    elif opts.action[0] == "test":
        if opts.model == Model.decision_tree:
            model = DecisionTreeClassifier.load(opts.action[1])
        else:
            raise ValueError(f"Unsupported model type {opts.model}")

        X_test, y_test = Preprocessor.access_data_labels(opts.action[2])
        model.print()
        predictions = model.predict(X_test)
        print("Accuracy is: " + str(accuracy_score(y_test, predictions)))

    else:
        raise ValueError(f"Unsupported action type {opts.action}")
