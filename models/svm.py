import sys
import numpy as np
from matplotlib import pyplot as plt

from preprocessing.preprocessor import Preprocessor


class SVM:
    """
    Support Vector Machine model trained through gradient ascent.
    """

    def __init__(self, lr=1e-3, max_iter=1000, kernel_type="rbf", c=1, sigma=1, d=3):
        """
        Initialize the SVM with the given configuration.
        :param lr: learning rate
        :param max_iter: max iterations to run gradient descent
        :param kernel_type: kernel type to be used
        :param c: hyperparameter determining "softness" of SVM
        :param sigma: hyperparameter determining level on non-linearity in RBF kernel
        :param d: hyperparameter determining dimension of polynomial regression
        """
        self.lr = lr
        self.max_iter = max_iter
        self.kernel = self._rbf_kernel if kernel_type == "rbf" else self._poly_kernel
        self.c = c
        self.sigma = sigma
        self.d = d
        # Generated during training
        self.x = None
        self.y = None
        self.alpha = None
        self.yyk = None

    def _rbf_kernel(self, x, y):
        """
        RBF Kernel function.
        :param x: nXm dataset of n entries, each with m features
        :param y: nXm dataset of n entries, each with m features
        :return: kernel mapping
        """
        return np.exp(
            -(1 / 2 * self.sigma ** 2) * np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :],
                                                        axis=2) ** 2)

    def _poly_kernel(self, x, y):
        """
        Polynomial Kernel function.
        :param x: nXm dataset of n entries, each with m features
        :param y: nXm dataset of n entries, each with m features
        :return: kernel mapping
        """
        return (np.inner(x, y) + self.c) ** self.d

    @staticmethod
    def _x_bar(x):
        """
        Transform the given vector of m freatures into a vectore of m + 1 features with
        preprended 1's.
        :param x: feature vector of length m
        :return: x_bar feature vector
        """
        return np.c_[np.ones(x.shape[0]), x]

    def _decision_function(self, x):
        """
        Access decision for the given features.  Assumes model has been trained.
        :param x: nXm dataset of n entries, each with m features
        :return: decisions
        """
        return (self.alpha * self.y).dot(self.kernel(self.x, x))

    def predict(self, x):
        """
        Access predictions for the given features.  Assumes model has been trained.
        :param x: feature vectors
        :return: 0 or 1 based on decision function result
        """
        return np.sign((self.alpha * self.y).dot(self.kernel(self.x, x)))

    def _compute_cost(self):
        """
        Compute the dual hinge loss.
        :return: cost
        """
        return np.sum(self.alpha) - (1 / 2) * np.sum(np.outer(self.alpha, self.alpha) * self.yyk)

    def train(self, input_var, label, print_iter=500):
        """
        Train the model using batch gradient ascent.
        :param input_var: nXm feature vectors
        :param label: n labels
        :param print_iter: log frequency
        :return: tuples of (costs, final alpha)
        """
        x_bar = SVM._x_bar(input_var)  # compute x_bar
        self.alpha = np.random.random(x_bar.shape[0])  # initialize random alpha
        ones = np.ones(x_bar.shape[0])
        self.x = x_bar
        self.y = label

        # precompute (y_j)(y_k)(K(x_j, x_k))
        self.yyk = np.outer(label, label) * self.kernel(self.x, self.x)

        costs = []
        iteration = 0
        while iteration < self.max_iter:

            if iteration % print_iter == 0:
                print(f'iteration: {iteration}')
                cost = self._compute_cost()
                costs.append(cost)
                print(f'cost: {cost}')
                print('--------------------------------------------')

            gradient = ones - self.yyk.dot(self.alpha)
            self.alpha += self.lr * gradient

            # Keep 0 <= alpha <= c
            self.alpha[self.alpha > self.c] = self.c
            self.alpha[self.alpha < 0] = 0

            iteration += 1

        return costs, self.alpha

    def test(self, input_test, label_test):
        """
        Test the accuracy of the model using the given test features and labels.
        :param input_test: test features
        :param label_test: test labels
        :return: accuracy as percentage
        """
        prediction = self.predict(SVM._x_bar(input_test))
        return np.mean(label_test == prediction)


if __name__ == '__main__':
    print(sys.argv)
    X_train, y_train = Preprocessor.access_data_labels(sys.argv[1])

    # Train to identify class 1 vs all
    y_train[y_train != 1] = -1

    svm = SVM(max_iter=0)

    # costs, alpha = svm.train(X_train, y_train)

    # plt.plot(costs)
    # plt.title("cost per epochs * pint_iter")
    # plt.show()
    #
    # X_test, y_test = Preprocessor.access_data_labels(sys.argv[2])
    # print(f'Accuracy {svm.test(X_test, y_test)}')
    #
    # # Plot points
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, alpha=.5)
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    #
    # # Create mesh
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = (alpha * y_train).dot(svm.kernel(X_train, xy)).reshape(XX.shape)
    #
    # # Plot boundaries
    # ax.contour(XX, YY, Z, colors=['b', 'g', 'r'], levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'], linewidths=[1.0, 1.0, 1.0])
    #
    # # Outline support vectors with nonzero alpha
    # ax.scatter(X_train[:, 0][alpha > 0.], X_train[:, 1][alpha > 0.],
    #            linewidth=1, facecolors='none', edgecolors='k')
    #
    # plt.show()
