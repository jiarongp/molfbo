from .sobol import Sobol
from botorch.utils.sampling import draw_sobol_samples
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel


class Sobol(Sobol):

    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

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

    def observe_and_suggest(self, X_obs, y_obs, cost_model, X_pen=None, batch_size=1, *args, **kwargs):
        """
        Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization of the qNParEGO
        acquisition function, and returns a new candidate and obervation.
        """
        new_x = draw_sobol_samples(bounds=self.problem.bounds, n=1, q=batch_size).squeeze(1)
        eval_objective_indices = list(range(len(cost_model.fixed_cost)))
        return new_x, eval_objective_indices
