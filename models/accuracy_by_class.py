import numpy as np


def accuracy(actual, predicted):
    tp = np.zeros(4)
    counts = np.zeros(4)
    gesture = ["Rock", "Scissors", "Paper", "OK"]

    for i in range(len(actual)):
        counts[int(actual[i])] += 1
        if actual[i] == predicted[i]:
            tp[int(actual[i])] += 1

    for i in range(4):
        print("Accuracy for " + str(i) + " (" + gesture[i] + ") is %.2f" % (tp[i] / counts[i] * 100))

    print("Overall accuracy is %.2f" % (np.sum(tp) / np.sum(counts * 100)))
