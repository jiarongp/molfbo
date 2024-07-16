from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import sample_simplex, draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from .acquisition_functions import TchebyshevThompsonSampling


class TS_TCH:

    def __init__(self, problem, device="cpu:0", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.n_candidates = min(5000, max(2000, 200 * problem.dim))

    def fit_model(self, X_obs, y_obs):
        # build a GP model scalarized objectives
        models = []
        for i in range(y_obs.shape[-1]):
            models.append(
                SingleTaskGP(
                    X_obs, y_obs[:, i:i+1], outcome_transform=Standardize(m=1)
                )
            )
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def get_chebyshev(self, weights, Y):
        if Y.shape[-2] == 1:
            # If there is only one observation, set the bounds to be
            # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
            # divide by zero
            Y_bounds = torch.cat([Y, Y + 1], dim=0)
        else:
            # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m
            Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

        def obj(Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
            Y_normalized = normalize(Y, Y_bounds)
            product = weights * Y_normalized
            # assume maximization
            return product.min(dim=-1).values
        return obj

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        """
        Samples a set of random weights for each candidate in the batch
        """
        assert batch_size == 1
        # normalize x
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # fit GP models for each objective
        self.fit_model(X_obs_norm, y_obs)

        # sample new weights
        weight = sample_simplex(d=self.problem.num_objectives, **self.tkwargs)
        thompson_sampling = TchebyshevThompsonSampling(
            model=self.model,
            objective=GenericMCObjective(self.get_chebyshev(weights=weight, Y=y_obs)),
            replacement=False
        )
        # generate candidate for maximization: n x q x d
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=self.n_candidates, q=1)
        # switch to q x n x d, since the q number doesn't support large number
        x_cands = x_cands.transpose(0, 1)
        with torch.no_grad():
            candidates = thompson_sampling(x_cands)

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x