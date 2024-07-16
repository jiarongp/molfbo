import torch
import pygmo as pg
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples

from .lfbo_joint_rand import LFBO_JointRand


def get_partial_observations(y_obs, gamma, tkwargs):
    ndf, _, _, _ = pg.core.fast_non_dominated_sorting(-y_obs.numpy())
    print(f"Number of pareto shell {len(ndf)}")

    num_obs = 0
    for n, shell in enumerate(ndf, start=1):
        num_obs += len(shell)
        if num_obs / len(y_obs) > gamma:
            break
    shell_idx = n

    y_shell = torch.empty(0, y_obs.shape[-1], **tkwargs)
    for i in range(shell_idx, len(ndf), 1):
        y_shell = torch.cat((y_shell, y_obs[ndf[i].astype(int)]))

    return y_shell


class MOLFBO_SHELL:

    def __init__(self, problem, device="cpu", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.n_candidates = min(5000, max(2000, 200 * problem.bounds.shape[-1]))
        self.warm_start = True
        self.clf_list = [
            LFBO_JointRand(input_dim=problem.bounds.shape[-1], output_dim=1, weight_type='pi', **self.tkwargs)
            for _ in range(problem.num_objectives)
        ]

    def fit_model(self, X_obs, y_obs, S):
        for i, clf in enumerate(self.clf_list):
            clf.fit(X_obs=X_obs, y_obs=y_obs[:, i:i+1], S=S)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)
        
        y_shell = get_partial_observations(y_obs, gamma=0.1, tkwargs=self.tkwargs)

        # they assumes maximization
        bd = DominatedPartitioning(ref_point=self.problem.ref_point, Y=y_shell)
        nbd = FastNondominatedPartitioning(ref_point=self.problem.ref_point, Y=y_shell)
        _, pareto = bd.hypercell_bounds
        ndom, _ = nbd.hypercell_bounds

        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=1, q=self.n_candidates).squeeze(0)
        # negate to turn into minimization
        y_obs = -y_obs
        pareto = -pareto
        ndom = -ndom

        if self.warm_start:
            self.warm_start = False
            S = 1000
        else:
            S = 100

        self.fit_model(X_obs_norm, y_obs, S=S)

        # u: upper non-dominated point
        # l: lower dominated point
        ref_pts = torch.concat((ndom, pareto), dim=0)
        pi_per_region = torch.empty((0, self.n_candidates, 1), **self.tkwargs)
        for ref in ref_pts:
            preds = torch.empty((0, self.n_candidates, 1), **self.tkwargs)
            for i, clf in enumerate(self.clf_list):
                gamma = ((y_obs[:, i:i+1] <= ref[i]).sum() / len(y_obs)).to(**self.tkwargs)

                with torch.no_grad():
                    preds = torch.concat([preds, clf.predict(x_cands, gamma=gamma).unsqueeze(0)], dim=0)

            agg_preds = torch.cumprod(preds, dim=0)[-1]
            pi_per_region = torch.concat((pi_per_region, agg_preds.unsqueeze(0)), dim=0)

        # the number pareto points is always one less than the non-dominated points 
        pi_per_region = torch.concat((pi_per_region, torch.zeros_like(pi_per_region[0]).unsqueeze(0)), dim=0)
        pi_per_interval = pi_per_region[:len(ndom)] - pi_per_region[len(ndom):]
        pi = torch.sum(pi_per_interval, dim=0)
        candidates = x_cands[pi.argmax()].unsqueeze(0)
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
