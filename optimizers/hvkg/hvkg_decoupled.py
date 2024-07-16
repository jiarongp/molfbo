import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel


class HVKG:

    def __init__(self, problem, device="cpu:0", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128

        self.num_pareto = 10
        self.num_fantasies = 32
        self.num_hvkg_restarts = 1

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

    def get_current_value(self):
        curr_val_acqf = _get_hv_value_function(
            model=self.model,
            ref_point=self.problem.ref_point,
            use_posterior_mean=True
        )
        _, curr_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.standard_bounds,
            q=self.num_pareto,
            num_restarts=20,
            raw_samples=1024,
            return_best_only=True,
            options={"batch_limit": 5}
        )
        return curr_value

    def observe_and_suggest(self, X_obs_list, y_obs_list, cost_model, X_pen=None, batch_size=1, *args, **kwargs):
        """
        optimize the qEHVI acquisition function, and returns a new candidate and obervation.
        """
        # normalize training input
        X_obs_norm_list = [normalize(X_obs, self.problem.bounds) for X_obs in X_obs_list]

        # update GP models for each objective
        self.fit_model(X_obs_norm_list, y_obs_list)

        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        # get the best ei point, i.e. the second term in kg acquisition function
        current_value = self.get_current_value()

        acq_func = qHypervolumeKnowledgeGradient(
            model=self.model,
            ref_point=self.problem.ref_point,
            num_fantasies=self.num_fantasies,
            num_pareto=self.num_pareto,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility
        )

        # optimize acquisition function and get new observation
        objective_vals = []
        objective_candidates = []
        # indices is set to the order indices of objective
        objective_indices = list(range(len(cost_model.fixed_cost)))
        for objective_idx in objective_indices:
            # set evaluation index to only conditioned on one objective
            # could be multiple objective
            X_evaluation_mask = torch.zeros(
                1,
                len(objective_indices),
                dtype=torch.bool,
                device=self.tkwargs['device']
            )
            X_evaluation_mask[0, objective_idx] = 1
            acq_func.X_evaluation_mask = X_evaluation_mask
            candidates, vals = optimize_acqf(
                acq_function=acq_func,
                num_restarts=self.num_hvkg_restarts,
                raw_samples=self.raw_samples,
                bounds=self.standard_bounds,
                q=batch_size,
                sequential=False,
                options={"batch_limit": 5},
            )
            objective_vals.append(vals.view(-1))
            objective_candidates.append(candidates)
        best_objective_index = torch.cat(objective_vals, dim=-1).argmax().item()
        eval_objective_indices = [best_objective_index]
        candidates = objective_candidates[best_objective_index]
        vals = objective_vals[best_objective_index]

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)

        return new_x, eval_objective_indices
