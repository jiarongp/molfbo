import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class qParEGO:

    def __init__(self, problem, device="cpu", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128

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
        Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization of the qNParEGO
        acquisition function, and returns a new candidate and obervation.
        """
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # fit GP models for each objective
        self.fit_model(X_obs_norm, y_obs)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        with torch.no_grad():
            pred = self.model.posterior(X_obs_norm).mean

        acq_func_list = []
        for _ in range(batch_size):
            weights = sample_simplex(self.problem.num_objectives, **self.tkwargs).squeeze()

            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qNoisyExpectedImprovement(
                model=self.model,
                objective=objective,
                X_baseline=X_obs_norm,
                sampler=sampler,
                prune_baseline=True
            )
            acq_func_list.append(acq_func)

        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.standard_bounds,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples, # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200}
        )

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
