import warnings
from typing import Dict, Any

import cma
import numpy as np
import torch
from scipy.stats.qmc import Sobol, scale
from scipy.optimize import minimize
from botorch.utils.sampling import draw_sobol_samples

from .classifiers import ClassifierBase


class OptimizerBase:
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2000,
    ) -> None:
        self.model = model
        self.lb = lb
        self.ub = ub
        self.budget = budget

    def suggest(self) -> np.ndarray:
        raise NotImplementedError


class RandomSearchOptimizer(OptimizerBase):
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2048,
        boltzman_sample: bool = True,
    ):
        super(RandomSearchOptimizer, self).__init__(model, lb, ub, budget)
        self.boltzman_sample = boltzman_sample
        self.sobol = Sobol(d=lb.size, scramble=True)

    def suggest(self):
        # sample from a scrambled sobol sequence in [0, 1]^d and
        # ignore warnings about not sampling a power of 2 samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_eval = self.sobol.random(n=self.budget)

        # rescale to original bounds
        X_eval = scale(X_eval, self.lb, self.ub)

        # evaluate with the model
        p = self.model.predict(X_eval)

        if self.boltzman_sample:
            eta = 1.0

            # standardize p first -- avoiding any tiny standard deviation
            pmean, pstd = np.mean(p), np.std(p)
            pstd = pstd if pstd > 1e-6 else 1.0
            p = (p - pmean) / pstd

            # also clip into the range we can exponentiate without failure.
            # note that mostly this won't do anything
            p = np.clip(p, -eta * 88.72, eta * 88.72)

            # temper the values and rescale so they add up to one
            p = np.exp(eta * p)
            p /= np.sum(p)

            # sample a location proportional to its tempered value
            new_candidate_idx = np.random.choice(self.budget, p=p, replace=False)

        # simply argmax
        else:
            # get the index/indices that are equal to the maximum
            argmax_locs = np.where(p == np.max(p))[0]

            # if there's more than one, choose randomly
            if len(argmax_locs) > 1:
                new_candidate_idx = np.random.choice(argmax_locs, replace=False)
            else:
                new_candidate_idx = argmax_locs[0]

        return X_eval[new_candidate_idx, :]


class CMAESOptimizer(OptimizerBase):
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2048,
        cma_options: Dict[str, Any] = {},
    ):
        super(CMAESOptimizer, self).__init__(model, lb, ub, budget)

        self.cma_options = {
            "bounds": [list(self.lb), list(self.ub)],
            "tolfun": 1e-7,
            "maxfevals": self.budget,
            "verb_log": 0,
            "verb_disp": 0,
            "verbose": -3,  # suppress any warnings (from flat fitness)
            "CMA_stds": np.abs(self.ub - self.lb),
        }

        self.cma_options.update(cma_options)

        self.cma_sigma = 0.25
        self.cma_centroid = lambda: np.random.uniform(lb, ub)

    def suggest(self):
        res = cma.fmin(
            None,  # no function as specifying a parallel one
            self.cma_centroid,  # random evaluation within bounds
            self.cma_sigma,
            options=self.cma_options,
            parallel_objective=self._cma_objective,
            args=(self.model,),
            bipop=True,
            restarts=10,
        )

        xnext = res[0]
        return xnext

    @staticmethod
    def _cma_objective(X, model):
        # turn the list of decision vectors into a numpy array
        X = np.stack(X)

        # evaluate the model
        fx = model.predict(X)

        # negate because CMA-ES minimises
        fx = -fx

        # convert to a list as this is what CMA-ES expects back
        fx = fx.tolist()

        return fx


def lbfgs_optimizer(
    model,
    bounds,
    optimization_budget,
    num_starts=10,
    method="L-BFGS-B",
    options=dict(maxiter=1000, ftol=1e-9),
):
    x_init = draw_sobol_samples(bounds=bounds, n=optimization_budget, q=1).squeeze(1)
    with torch.no_grad():
        pred_logits = model(x_init)
    dr_mdre = pred_logits[:, 0] - pred_logits[:, 1]
    # the function to minimize is negative of the classifier output
    f_init = -dr_mdre.detach().cpu().numpy()

    def objective(x):
        x = torch.tensor(x, requires_grad=True)
        pred_logits = model(x)
        dr_mdre = pred_logits[0] - pred_logits[1]
        y = - dr_mdre
        y.backward()
        return y.detach().cpu().tolist(), x.grad.detach().cpu().tolist()

    x_cands = []
    y_cands = []
    ind = np.argpartition(f_init, kth=num_starts-1, axis=None)
    for i in range(num_starts):
        x0 = x_init[ind[i]]
        result = minimize(
            objective,
            x0=x0,
            method=method,
            jac=True,
            bounds=bounds.t().tolist(),
            options=options
        )
        x_cands.append(result.x)
        y_cands.append(result.fun)

    return x_cands[np.argmin(y_cands)]


def logistic_lbfgs_optimizer(
    model,
    bounds,
    optimization_budget,
    num_starts=10,
    method="L-BFGS-B",
    options=dict(maxiter=1000, ftol=1e-9),
):
    x_init = draw_sobol_samples(bounds=bounds, n=optimization_budget, q=1).squeeze(1)
    with torch.no_grad():
        _, pred_logits = model(x_init)
    # the function to minimize is negative of the classifier output
    f_init = -pred_logits.detach().cpu().numpy()

    def objective(x):
        x = torch.tensor(x.reshape(1, -1), requires_grad=True)
        _, pred_logits = model(x)
        y = -pred_logits
        y.backward()
        return y.detach().cpu().tolist(), x.grad.detach().cpu().tolist()

    x_cands = []
    y_cands = []
    ind = np.argpartition(f_init, kth=num_starts-1, axis=None)
    for i in range(num_starts):
        x0 = x_init[ind[i]]
        result = minimize(
            objective,
            x0=x0,
            method=method,
            jac=True,
            bounds=bounds.t().tolist(),
            options=options
        )
        x_cands.append(result.x)
        y_cands.append(result.fun)

    return x_cands[np.argmin(y_cands)]