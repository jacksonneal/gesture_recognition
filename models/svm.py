class SVM:
    """
    Support Vector Model trained through gradient descent.
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
