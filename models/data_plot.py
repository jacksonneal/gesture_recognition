import matplotlib.pyplot as plt
import numpy as np

from models.bayes import NaiveBayesClassifier

if __name__ == '__main__':

    classifier = NaiveBayesClassifier.load("data.txt")
    # model[target]: mean, stdev, count
    mean_col = 1
    stdev_col = 2
    rock = np.array(classifier.model[0])[:, mean_col]
    r_std = np.array(classifier.model[0])[:, stdev_col]

    scissors = np.array(classifier.model[1])[:, mean_col]
    s_std = np.array(classifier.model[1])[:, stdev_col]

    paper = np.array(classifier.model[2])[:, mean_col]
    p_std = np.array(classifier.model[2])[:, stdev_col]

    ok = np.array(classifier.model[3])[:, mean_col]
    o_std = np.array(classifier.model[3])[:, stdev_col]

    colors = ('orange', 'blue', 'green', 'red')

    plt.plot(rock, label = "Rock", color=colors[0])
    plt.plot(scissors, label = "Scissors", color=colors[1])
    #plt.plot(paper, label = "Paper", color=colors[2])
    #plt.plot(ok, label = "OK", color=colors[3])

    a = 0.2
    scale = 0.3
    plt.fill_between(range(64), rock - scale*r_std, rock + scale*r_std, alpha=a, color=colors[0])
    plt.fill_between(range(64), scissors - scale*s_std, scissors + scale*s_std, alpha=a, color=colors[1])
    #plt.fill_between(range(64), paper - scale*p_std, paper + scale*p_std, alpha=a, color=colors[2])
    #plt.fill_between(range(64), ok - scale*o_std, ok + scale*o_std, alpha=a, color=colors[3])

    plt.legend()
    plt.savefig("..\\graphs\\Rock_Scissors.png", format="png")
    plt.show()
