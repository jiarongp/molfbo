import torch
import numpy as np
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import unnormalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch import settings
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination


def get_model_identified_hv_maximizing_set(
    problem,
    model,
    population_size=250,
    max_gen=100,
):
    """Optimize the posterior mean using NSGA-II."""
    tkwargs = {
        "dtype": problem.ref_point.dtype,
        "device": problem.ref_point.device,
    }
    dim = problem.dim

    class PosteriorMeanPymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=dim,
                n_obj=problem.num_objectives,
                type_var=np.double,
            )
            self.xl = np.zeros(dim)
            self.xu = np.ones(dim)

        def _evaluate(self, x, out, *args, **kwargs):
            X = torch.from_numpy(x).to(**tkwargs)
            is_fantasy_model = (
                isinstance(model, ModelListGP)
                and model.models[0].train_targets.ndim > 2
            ) or (
                not isinstance(model, ModelListGP) and model.train_targets.ndim > 2
            )
            with torch.no_grad():
                with settings.cholesky_max_tries(9):
                    # eval in batch mode
                    y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                if is_fantasy_model:
                    y = y.mean(dim=-2)
            out["F"] = -y.cpu().numpy()

    pymoo_problem = PosteriorMeanPymooProblem()
    algorithm = NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
    )
    res = minimize(
        pymoo_problem,
        algorithm,
        termination=MaximumGenerationTermination(max_gen),
        # seed=0,  # fix seed
        verbose=False,
    )
    X = torch.tensor(
        res.X,
        **tkwargs,
    )
    X = unnormalize(X, problem.bounds)
    Y = problem(X)
    # compute HV
    partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point, Y=Y)
    return partitioning.compute_hypervolume().item()
