import numpy as np


def accuracy(actual, predicted):
    tp = np.zeros(4)
    counts = np.zeros(4)

    for i in range(len(actual)):
        counts[int(actual[i])] += 1
        if actual[i] == predicted[i]:
            tp[int(actual[i])] += 1

    for i in range(4):
        print("Accuracy for " + str(i) + " is " + str(tp[i] / counts[i]))

    print("Overall accuracy is " + str(np.sum(tp) / np.sum(counts)))
