import torch
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from optimizers.pfns.scripts.acquisition_functions import general_acq_function, optimize_acq_w_lbfgs, optimize_acq


class PFNs:

    def __init__(
        self,
        problem,
        acq_f=general_acq_function,
        fit_encoder=None,
        optimize_acq='lbfgs',
        device="cpu:0",
        dtype=torch.double,
    ):

        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128

        # PFNs parameters
        hebo_plus_model = "optimizers/pfns/final_models/model_hebo_morebudget_9_unused_features_3_whole_model.pt"
        self.model = torch.load(hebo_plus_model)
        self.device = device
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder
        self.optimize_acq = optimize_acq

    @staticmethod
    def normalize_y(y_obs):
        y_mean = y_obs.mean()
        y_std = y_obs.std()
        return (y_obs - y_mean) / y_std

    @staticmethod
    def unnormlize_y(y_obs, y_data):
        y_mean = y_obs.mean()
        y_std = y_obs.std()
        return y_data * y_std + y_mean
    
    @staticmethod
    def chebyshev_obj(weights, Y, alpha=0.05):
        Y = -Y
        product = weights * Y
        return product.max(dim=-1).values + alpha * product.sum(dim=-1)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1):
        """
        Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization of the qNParEGO
        acquisition function, and returns a new candidate and obervation.
        """
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)

        self.model.to(**self.tkwargs)

        # fine-tuning
        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        # if X_pen is None:
        #     X_pen = torch.rand(X_obs_norm.shape[-1], **self.tkwargs).reshape(1, -1)

        with torch.no_grad():
            # pred = torch.tensor([], **self.tkwargs)
            pred = torch.empty(X_obs_norm.shape[0], 0, **self.tkwargs)
            for i in range(y_obs.shape[-1]):
                pred_per_obj = self.acq_function(
                    self.model,
                    X_obs_norm, y_obs[:, i:i+1], X_obs_norm,
                    acq_function='mean', apply_power_transform=True, return_actual_ei=True,
                    # normalize y seems have better results
                    znormalize=False,
                )
                # pred_per_obj = self.unnormlize_y(y_obs[:, i:i+1], pred_per_obj)
                # use normalized y
                pred = torch.concat([pred, pred_per_obj.reshape(-1, 1)], dim=1)

        candidates = torch.empty(0, X_obs_norm.shape[1], **self.tkwargs)
        for _ in range(batch_size):
            weights = sample_simplex(self.problem.num_objectives, **self.tkwargs).squeeze()
            y_scalarized = -self.chebyshev_obj(weights=weights, Y=pred)

            with torch.enable_grad():
                if self.optimize_acq == 'adam':
                    new_candidate = optimize_acq(
                        self.model, X_obs_norm, y_scalarized, num_grad_steps=10, num_random_samples=100, lr=.01,
                        znormalize=False, apply_power_transform=True,
                    )
                elif self.optimize_acq == 'lbfgs':
                    new_candidate, x_options, eis, x_rs, x_rs_eis  = optimize_acq_w_lbfgs(
                        self.model, X_obs_norm, y_scalarized,
                        num_grad_steps=15000, num_candidates=10, pre_sample_size=512,
                        verbose=False, znormalize=False, apply_power_transform=True, **self.tkwargs
                    )
            new_candidate = new_candidate.detach().cpu()
            candidates = torch.concat([candidates, new_candidate.reshape(1, -1)], dim=0)

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
