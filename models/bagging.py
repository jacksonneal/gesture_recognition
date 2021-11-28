import json
import random
import pandas as pd
import numpy as np

from models.decision_tree import DecisionTreeClassifier
from models.model import Algo, Model


class Bagging(Algo):
    """
    Bagging Ensemble model.
    """

    def __init__(self, algos, k):
        """
        Construct an Bagging ensemble model with the given models.
        :param algos: underlying models of the ensemble
        :param k: number of entries to sample from dataset using bootstrapping and use to train
        each model with
        """
        self.algos = algos
        self.k = k

    def print(self):
        print("Bagging:")
        for algo in self.algos:
            algo.print()

    def save(self, dest):
        obj = {
            "type": Model.bagging,
            "algos": self.algos,
            "k": self.k
        }
        with open(dest, "w") as f:
            json.dump(obj, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def load(src):
        with open(src, "r") as f:
            return json.load(f, object_hook=Bagging.as_payload)

    @staticmethod
    def as_payload(dct):
        if dct["type"] == Model.bagging:
            return Bagging(dct["algos"], dct["k"])
        elif dct["type"] == Model.decision_tree:
            return DecisionTreeClassifier(dct["min_samples_split"], dct["max_depth"],
                                          dct["max_split_eval"], dct["root"], dct["mode"] == "gini")
        else:
            raise ValueError("Unsupported model type.")

    def fit(self, X, Y):
        dataset = np.insert(X, -1, Y, axis=1)
        for algo in self.algos:
            bootstrap_sample = pd.Dataframe([random.choice(dataset) for _ in range(self.k)])
            x, y = bootstrap_sample.iloc[:, :-1].values, bootstrap_sample.iloc[:, -1] \
                .values.reshape(-1, 1)
            algo.fit(x, y)

    def predict(self, X):
        predictions = {}
        for algo in self.algos:
            prediction = algo.predict(X)
            if prediction not in predictions:
                predictions[prediction] = 0
            predictions[prediction] += 1
        return max(predictions, key=predictions.get)
