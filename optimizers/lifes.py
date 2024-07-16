import numpy as np
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity

from .utils import split_good_bad
from .bore import BORE


class LIFES:
    def __init__(self, obj_func, l_bounds, u_bounds) -> None:
        self.obj_func = obj_func
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

    def compute_entropy(self, X, log_p):
        log_prob = log_p(X)
        prob = np.exp(log_prob)
        return - np.mean(prob * log_prob)

    def compute_expected_entropy(
        self,
        data,
        x_sample,
        qmc_samples,
        optimizer
    ):
        X_obs, y_obs, z_obs = data
        
        x_sample = x_sample.reshape(-1, 1)
        new_X_obs = np.concatenate((X_obs, x_sample))
        # we still need actual y value to fit density
        sample_y = self.obj_func(x_sample, noise=True)
        new_y_obs = np.concatenate((y_obs, sample_y.reshape(-1, 1)))
        new_z_obs_pos = np.concatenate((z_obs, [True]))
        new_z_obs_neg = np.concatenate((z_obs, [False]))

        optimizer_pos = KernelDensity(kernel='gaussian', bandwidth=.2)
        optimizer_neg = KernelDensity(kernel='gaussian', bandwidth=.2)

        optimizer_pos.fit(new_X_obs[new_z_obs_pos], new_y_obs[new_z_obs_pos])
        optimizer_neg.fit(new_X_obs[new_z_obs_neg], new_y_obs[new_z_obs_neg])

        e_pos = self.compute_entropy(
            qmc_samples,
            optimizer_pos.score_samples
        )
        e_neg = self.compute_entropy(
            qmc_samples,
            optimizer_neg.score_samples
        )
        prob = optimizer.predict(x_sample)

        return e_neg + prob * (e_pos - e_neg)

    def _entropy_search(self, X, data, gamma=0.3):
        sampler = qmc.Sobol(d=1)
        sample_continued = sampler.random_base2(m=10)
        qmc_samples = qmc.scale(
            sample_continued,
            self.l_bounds,
            self.u_bounds
        )

        X_obs, y_obs, z_obs = split_good_bad(*data, gamma=gamma)
        optimizer_bore = BORE()
        optimizer_bore.fit(X_obs, y_obs)

        optimizer_tpe = KernelDensity(kernel='gaussian', bandwidth=.2)
        optimizer_tpe.fit(*data)
        entropy = self.compute_entropy(
            qmc_samples,
            optimizer_tpe.score_samples
        )
        
        acf = []
        for x_sample in X:
            expected_entropy = self.compute_expected_entropy(
                (X_obs, y_obs, z_obs),
                x_sample,
                qmc_samples,
                optimizer_bore
            )
            acf_value = entropy - expected_entropy
            acf.append(acf_value)
            
        return np.array(acf)

    def predict(self, X, data, gamma=0.3):
        return self._entropy_search(X, data, gamma)
