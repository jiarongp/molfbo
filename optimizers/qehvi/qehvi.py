import time

import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from utils.helper import configure_logger


class qEHVI:

    def __init__(self, problem, device="cpu", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128
        self.logger = configure_logger(__name__)
        self.logger.info("qEHVI")

    def fit_model(self, X_obs, y_obs):
        # define model for objective and constraint
        models = []
        # build a GP model for each objective
        for i in range(y_obs.shape[-1]):
            models.append(
                SingleTaskGP(
                    X_obs, y_obs[..., i:i+1], outcome_transform=Standardize(m=1)
                )
            )
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        """
        optimize the qEHVI acquisition function, and returns a new candidate and obervation.
        """
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # # update GP models for each objective
        fit_time = time.time()
        self.fit_model(X_obs_norm, y_obs)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        with torch.no_grad():
            pred = self.model.posterior(X_obs_norm).mean

        # partition non-dominated space into disjoint rectangles
        partitioning = FastNondominatedPartitioning(
            ref_point=self.problem.ref_point,
            Y=pred
        )

        self.acq_func = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.problem.ref_point,
            partitioning=partitioning,
            sampler=sampler
        )

        # optimize
        opt_time = time.time()
        candidates, _ = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.standard_bounds,
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples, # used for intialization heuristic
            options={"batch_limit": 6, "maxiter": 200},
            sequential=True,
        )
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
