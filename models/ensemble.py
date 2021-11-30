import json
import os.path
import random
import pandas as pd
import numpy as np

from decision_tree import DecisionNode, DecisionTreeClassifier, LeafNode
# from models.decision_tree import DecisionNode, DecisionTreeClassifier, LeafNode
from models.bayes import NaiveBayesClassifier
from models.model import Algo, Model

MODEL_OFFSET = 0  # Naming offset for training ensemble in parts
MAX_BOOST = 100  # maximum number of samples to boost


class Ensemble(Algo):
    """
    Bagging Ensemble model.
    """

    def __init__(self, algos, k, boosting=False, num_models=None):
        """
        Construct an Bagging ensemble model with the given models.
        :param algos: underlying models of the ensemble
        :param k: number of entries to sample from dataset using bootstrapping and use to train
        :param boosting: whether to use boosting in training
        :param num_models: number of expected models
        """
        self.algos = algos
        self.k = k
        self.num_models = len(algos) if num_models is None else num_models
        self.boosting = boosting

    def debug_print(self):
        print("Bagging:")
        for algo in self.algos:
            algo.print()

    def save(self, dest):
        obj = {
            "type": Model.ensemble.value,
            "boosting": self.boosting,
            "num_models": self.num_models,
            "k": self.k
        }
        with open(os.path.join(dest, "bag.json"), "w") as f:
            json.dump(obj, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        for i, algo in enumerate(self.algos):
            algo.save(os.path.join(dest, str(i + MODEL_OFFSET) + ".json"))

    @staticmethod
    def load(src):
        with open(os.path.join(src, "bag.json"), "r") as f:
            bag = json.load(f, object_hook=Ensemble.as_payload)
        for i in range(bag.num_models):
            with open(os.path.join(src, str(i) + ".json")) as f:
                model = json.load(f, object_hook=Ensemble.as_payload)
                bag.algos.append(model)
            # print(f"Loaded model: {i}")
        return bag

    @staticmethod
    def as_payload(dct):
        if "value" in dct:
            return LeafNode(dct["value"])
        elif "feature_index" in dct:
            return DecisionNode(dct["feature_index"], dct["threshold"],
                                dct["left"], dct["right"],
                                dct["info_gain"])
        elif dct["type"] == Model.ensemble.value:
            return Ensemble([], dct["k"], dct["boosting"], dct["num_models"])
        elif dct["type"] == Model.decision_tree.value:
            return DecisionTreeClassifier(dct["min_samples_split"], dct["max_depth"],
                                          dct["max_split_eval"], dct["root"], dct["mode"] == "gini",
                                          dct["num_valid_features"])
        elif dct["type"] == Model.bayes.value:
            # We just assume we have a bayes
            loaded_classifier = NaiveBayesClassifier()
            loaded_classifier.training_data_len = 0
            loaded_classifier.model = {}

            for key in dct:
                if key == "type":
                    continue
                loaded_classifier.model[float(key)] = dct[key]
                loaded_classifier.training_data_len += dct[key][0][2]

            return loaded_classifier
        else:
            raise ValueError(f"Unsupported model type {dct['type']}.")

    def fit(self, X, Y):
        dataset = np.append(X, Y, axis=1)
        for i, algo in enumerate(self.algos):
            bootstrap_sample = pd.DataFrame([random.choice(dataset) for _ in range(self.k)])
            x, y = bootstrap_sample.iloc[:, :-1].values, bootstrap_sample.iloc[:, -1] \
                .values.reshape(-1, 1)
            algo.fit(x, y)
            if self.boosting:
                failed = []
                for row in dataset:
                    x, y = row[:-1], row[-1]
                    if algo.predict([x])[0] != y:
                        failed.append(row)
                if len(failed) > MAX_BOOST:
                    failed = random.sample(failed, MAX_BOOST)
                for row in failed:
                    dataset = np.vstack([dataset, row])

            print(f"Trained model: {i}")

    def predict(self, X):
        ret = []
        for x in X:
            predictions = {}
            for algo in self.algos:
                prediction = algo.predict([x])[0]
                if prediction not in predictions:
                    predictions[prediction] = 0
                predictions[prediction] += 1
            ret.append(max(predictions, key=predictions.get))
        return ret
