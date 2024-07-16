import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood


class ParEGO:

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
        Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization of the qNParEGO
        acquisition function, and returns a new candidate and obervation.
        """
        if batch_size > 1:
            raise ValueError(f"not supported for batch size larger than 1")

        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # scalarized objectives
        weights = sample_simplex(self.problem.num_objectives, **self.tkwargs).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=y_obs)
        )
        y_scalarized = objective(y_obs).reshape(-1, 1)

        # update GP models for each objective
        self.fit_model(X_obs_norm, y_scalarized)

        # define acquisition function
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=y_scalarized.max()
        )

        # optimize
        candidate, _ = optimize_acqf(
            acq_func,
            bounds=self.standard_bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        new_x = unnormalize(candidate.detach(), bounds=self.problem.bounds)
        return new_x
