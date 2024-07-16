import numpy as np
from sklearn.neighbors import KernelDensity

from .utils import split_good_bad

class TPE:
    def __init__(
        self,
        top_n_percent=30,
        bandwidth="silverman",
        algorithm="auto",
        kernel="gaussian",
        metric="euclidean",
    ):
        self.X = None
        self.y = None

        self.gamma = top_n_percent / 100
        self.bandwidth=bandwidth
        self.algorithm=algorithm
        self.kernel=kernel
        self.metric=metric
        
        self.kde_good = None
        self.kde_bad = None
 
    def fit(self, X, y, z=None):
        self.kde_good = KernelDensity(
            bandwidth=self.bandwidth,
            algorithm=self.algorithm,
            kernel=self.kernel,
            metric=self.metric,
        )

        self.kde_bad = KernelDensity(
            bandwidth=self.bandwidth,
            algorithm=self.algorithm,
            kernel=self.kernel,
            metric=self.metric,
        )

        if z is None:
            _, _, z = split_good_bad(X, y, gamma=self.gamma)

        self.kde_good.fit(X[z], y[z])
        self.kde_bad.fit(X[~z], y[~z])

    def predict(self, X):
        # We compute the gamma relative density ratio instead of just density ratio
        # becasue in the original TPE algorithm
        # $EI \propto (\gamma + (1-\gamma)g(x)/l(x))^{-1} $
        # it optimizes l(x) / g(x) only because the location of the optimum doesn't change
        # with \gamma being constant
        # However, the function changes if we want to use EI/PI as a indicator to p(x_* | D) 
        log_density_ratio = self.kde_good.score_samples(X) - self.kde_bad.score_samples(X)
        density_ratio = np.exp(log_density_ratio)
        # gamma_relative_density_ratio = 1 / (self.gamma + (1 - self.gamma) / density_ratio)
        return density_ratio
