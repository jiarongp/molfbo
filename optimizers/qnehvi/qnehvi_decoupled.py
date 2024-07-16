import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel


class qNEHVI:

    def __init__(self, problem, device="cpu:0", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128

    def fit_model(self, X_obs_list, y_obs_list):
        # define model for objective and constraint
        models = []
        # build a GP model for each objective
        for i in range(len(y_obs_list)):
            models.append(
                SingleTaskGP(
                    X_obs_list[i],
                    y_obs_list[i],
                    outcome_transform=Standardize(m=1),
                    covar_module=ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            ard_num_dims=X_obs_list[0].shape[-1],
                            lengthscale_prior=GammaPrior(2.0, 2.0),
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    )
                )
            )
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def observe_and_suggest(
        self,
        X_obs_list,
        y_obs_list,
        cost_model,
        X_pen=None,
        batch_size=1,
        *args, **kwargs
    ):
        """
        optimize the qNEHVI acquisition function, and returns a new candidate and obervation.
        """
        # normalize training input
        X_obs_norm_list = [normalize(X_obs, self.problem.bounds) for X_obs in X_obs_list]

        # # update GP models for each objective
        self.fit_model(X_obs_norm_list, y_obs_list)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.problem.ref_point.tolist(),
            X_baseline=X_obs_norm_list[0], # qNEHVI only works in coupled setting
            prune_baseline=True, # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler
        )

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples, # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        # assume to evaluate all objectives
        eval_objective_indices = list(range(len(cost_model.fixed_cost)))
        return new_x, eval_objective_indices
