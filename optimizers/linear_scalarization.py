import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood


class LinearScalarization:

    def __init__(self, problem, device="cpu:0", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512

    def fit_model(self, X_obs, y_obs):
        # build a GP model scalarized objectives
        self.model = SingleTaskGP(
            X_obs, y_obs, outcome_transform=Standardize(m=1)
        )
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1):
        """
        Samples a set of random weights for each candidate in the batch
        """
        # normalize x
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # linear scalarization
        weight = sample_simplex(d=self.problem.num_objectives, **self.tkwargs)
        y_scalarized = torch.matmul(y_obs, weight.t())

        # update GP models for each objective
        self.fit_model(X_obs_norm, y_scalarized)

        acq_func = ExpectedImprovement(
            self.model,
            best_f=y_scalarized.max()
        )

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples, # used for intialization heuristic
        )

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x