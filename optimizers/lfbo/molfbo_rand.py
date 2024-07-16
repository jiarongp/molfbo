import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples, sample_hypersphere

from .lfbo_joint_rand import LFBO_JointRand


class MOLFBO_Rand:

    def __init__(self, problem, device="cpu:0", dtype=torch.double):
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

    def random_hypervolume(self, y, hv_weights):
        # hypervolume scalarization
        k = y.shape[-1]
        c_k = torch.pow(torch.pi, k / 2) / (torch.pow(torch.tensor(2), k) * torch.lgamma(k/2 + 1).exp())
        scalar = ((y - self.problem.ref_point).clamp_min(0).unsqueeze(-3) / hv_weights).amin(dim=-1).pow(k).amax(dim=-1)
        hv_scalar = c_k * scalar.mean()
        return hv_scalar

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)


        hv_weights = sample_hypersphere(d=y_obs.shape[-1], n=100, qmc=True).abs().unsqueeze(1).to(**self.tkwargs)
        self.fit_model(X_obs_norm, y_obs)
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=1, q=self.n_candidates).squeeze(0)

        bd = FastNondominatedPartitioning(ref_point=self.problem.ref_point, Y=y_obs)
        u, _ = bd.hypercell_bounds

        candidate_x = torch.empty((0, x_cands.shape[1]), **self.tkwargs)
        candidate_vals = torch.tensor([], **self.tkwargs)
        for ref in u:
            preds = torch.empty((0, x_cands.shape[0]), **self.tkwargs)
            for i, clf in enumerate(self.clf_list):
                gamma = ((y_obs[:, i:i+1] >= ref[0]).sum() / len(y_obs)).to(**self.tkwargs)

                with torch.no_grad():
                    preds = torch.concat([preds, clf.predict(x_cands, gamma=gamma).unsqueeze(0)], dim=0)

            agg_preds = torch.cumprod(preds, dim=0)[-1]
            val, ind = agg_preds.max(dim=0)
            candidate_x = torch.concat([candidate_x, x_cands[ind].unsqueeze(0)], dim=0)
            candidate_vals = torch.concat([candidate_vals, val.unsqueeze(0)])

        candidates = candidate_x[candidate_vals.argmax()].unsqueeze(0)
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x