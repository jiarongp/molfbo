from typing import Optional

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification


def split_good_bad(X, y, gamma):
    tau = np.quantile(np.unique(y), q=gamma)
    z = np.less(y, tau)

    if len(X) > 1:
        z_idx = z.squeeze()

        x1, z1 = X[z_idx], z[z_idx]
        x0, z0 = X, np.zeros_like(z)

        w1 = (tau - y)[z_idx]
        # sometimes w1 is empty
        w1 = w1 / np.mean(w1) if len(w1) else w1
        w0 = 1 - z0

        x = np.concatenate([x1, x0], axis=0)
        z = np.concatenate([z1, z0], axis=0)
        s1 = x1.shape[0]
        s0 = x0.shape[0]

        w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
        w = w / np.mean(w)

    elif len(X) == 1:
        x = X
        w = np.ones_like(z)

    return x, y, z, w


class EvaluationSpecificationSampler:
    def __init__(self, search_space: ps.ParameterSpace, seed: Optional[int] = None):
        self.search_space = search_space.copy()
        self.seed = seed

        self.search_space.seed(seed)

    def __call__(self) -> EvaluationSpecification:
        pass

    def digest_evaluation(self, evaluation: Evaluation):
        pass


class RandomSearchSampler(EvaluationSpecificationSampler):
    def __init__(self, search_space: ps.ParameterSpace, seed: Optional[int] = None):
        super().__init__(search_space=search_space, seed=seed)

    def __call__(self) -> EvaluationSpecification:
        return EvaluationSpecification(configuration=self.search_space.sample())