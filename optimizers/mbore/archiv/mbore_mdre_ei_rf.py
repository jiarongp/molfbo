import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from sklearn.ensemble import RandomForestClassifier

from utils.helper import configure_logger
from .scalarizers import HypervoumeContribution


class MBORE_MDRE_EI_RF:

    def __init__(
        self,
        problem,
        acq_type='ei',
        device="cpu:0",
        dtype=torch.double
    ) -> None:
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128
        self.n_candidates = min(5000, max(2000, 200 * problem.dim))

        # budget for optimising model (acq func equivalent)# budget for optimising model (acq func equivalent)
        self.n_dim = problem.bounds.shape[-1]
        self.optimisation_budget = 1024 * self.n_dim
        self.acq_type = acq_type # "ei" or "pi"

        # "HypI", "DomRank", "HypCont", "ParEGO"
        self.scalariser = HypervoumeContribution(reference_point=-problem.ref_point.cpu().numpy())
        self.logger = configure_logger(__name__)
        self.logger.info(__name__)
        self.x_m = None

    def check_input_dims(self, X):
        # check if X has batch dimension
        if len(X.shape) == 1:
            X = X[..., None, None]
        elif len(X.shape) == 2:
            X = X[:, None, :]
        # X should have: num_obs x n_dim x 1
        return X

    @staticmethod
    def objective_thresholding(y_obs, ref_point, offset=1e-9):
        if len(ref_point.shape) == 1:
            ref_point = ref_point.reshape(1, -1)
        y_threshold = np.repeat(ref_point - offset, y_obs.shape[0], axis=0)
        better_than_ref = (y_obs < ref_point).all(axis=-1)
        y_threshold[better_than_ref] = y_obs[better_than_ref]
        return y_threshold

    def train_auxiliary_classifier(self, X):
        x_m = draw_sobol_samples(bounds=self.standard_bounds, n=len(X), q=1).squeeze(1).numpy()
        x = np.concatenate([X, x_m], axis=0)
        z_p = np.zeros(X.shape[0])
        z_m = np.ones(x_m.shape[0])
        z = np.concatenate([z_p, z_m], axis=0)
        auxi_clf = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
        auxi_clf.fit(x, z)
        return auxi_clf

    def generate_mdre_dataset(self, X, y, gamma, acq_type='ei'):
        # self.auxi_clf = self.train_auxiliary_classifier(X)

        tau = np.quantile(y, q=gamma)
        z = np.less(y, tau)

        if len(X) > 1 and acq_type == 'ei':
            x_p, y_p = X[z.squeeze()], y[z.squeeze()]
            w_p = (tau - y)[z]
            w_p = w_p / np.mean(w_p)

            x_q, y_q = X, y
            x_m = draw_sobol_samples(self.standard_bounds, n=len(X), q=1).squeeze(1)
            self.x_m = x_m
            x_m = x_m.numpy()
            x = np.concatenate([x_p, x_q, x_m], axis=0)

            z_p = np.zeros(x_p.shape[0])
            z_q = np.ones(x_q.shape[0])
            z_m = np.ones(x_m.shape[0]) * 2
            z = np.concatenate([z_p, z_q, z_m], axis=0)

            s_p = x_p.shape[0]
            s_q = x_q.shape[0]
            s_m = x_m.shape[0]
            w_p = w_p * (s_p + s_q + s_m) / s_p
            w_q = z_q * (s_p + s_q + s_m) / s_q

            # probs = self.auxi_clf.predict_proba(x_m)
            # w_m = np.clip(probs[:, 1] - 0.5, a_min=0, a_max=None) * 2 * (s_q + s_p + s_m) / s_m
            w_m = np.ones(x_m.shape[0]) * (s_q + s_p + s_m) / s_m
            
            w = np.concatenate([w_p, w_q, w_m], axis=0)
            w = w / w.mean()

            return x, z, w

        elif len(X) == 1 or acq_type == 'pi':
            z_idx = z.squeeze()

            x_p = X[z_idx]
            x_q = X[~z_idx]
            # squeeze the q dimension 
            x_m = draw_sobol_samples(bounds=self.problem.bounds, n=1024, q=1).squeeze(1)
            # x_m = draw_sobol_samples(bounds=self.problem.bounds, n=int(len(X)/2), q=1).squeeze(1)
            x = torch.cat([x_p, x_q, x_m], axis=0)
            
            z_p = torch.empty(x_p.shape[0], dtype=torch.long).fill_(0)
            z_q = torch.empty(x_q.shape[0], dtype=torch.long).fill_(1)
            z_m = torch.empty(x_m.shape[0], dtype=torch.long).fill_(2)
            z = torch.cat([z_p, z_q, z_m], axis=0)
            z_onehot = torch.nn.functional.one_hot(z).to(X.dtype)

            class_weight = torch.tensor([x_p.shape[0], x_q.shape[0], x_m.shape[0]]) / len(x)
            w_p = torch.empty(x_p.shape[0], 1, dtype=X.dtype).fill_(class_weight[0])
            w_q = torch.empty(x_q.shape[0], 1, dtype=X.dtype).fill_(class_weight[1])
            w_m = torch.empty(x_m.shape[0], 1, dtype=X.dtype).fill_(class_weight[2])
            w = torch.cat([w_p, w_q, w_m], axis=0)
            
            return x, z_onehot, w, class_weight

    def fit_model(self, X_obs, y_obs, gamma):
        x, z, w = self.generate_mdre_dataset(
            X_obs, y_obs, gamma=gamma, acq_type=self.acq_type
        )
        self.clf = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
        self.clf.fit(x, z, sample_weight=w)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, gamma=1/3, *args, **kwargs):
        """MBORE assumes minimization"""
        assert batch_size == 1, (f"MBORE only supports a single "
                                 f"batch dimension, but got {batch_size} "
                                 f"batch dimensions.")

        # mbore is for minimization, multiply by -1 to turn it into minimization
        y_obs = - y_obs
        # rescale decision vectors to [0, 1] and declare the associated bounds
        X_obs_norm = normalize(X_obs, self.problem.bounds)
        # slb, sub = np.zeros(self.n_dim), np.ones(self.n_dim)
        X_obs_norm = X_obs_norm.detach().cpu().numpy()
        y_obs = y_obs.detach().cpu().numpy()

        # NOTE: objective thresholding, this is a modification due to the ref_point setting in BoTorch
        y_obs = self.objective_thresholding(y_obs, ref_point=-self.problem.ref_point.numpy())

        # scalarise the objective values
        _, y_scalarized = self.scalariser.get_ranks(y_obs, return_scalers=True)

        # negate the scalarized since we want to maximize the hypervolume contribution
        # therefore minimize its negative

        fit_time = time.time()
        self.fit_model(X_obs_norm, -y_scalarized, gamma=gamma, *args, **kwargs)
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        # generate candidate for maximization: n x q x d
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=self.n_candidates, q=1).squeeze(1).numpy()
        opt_time = time.time()
        preds_prob = np.clip(self.clf.predict_proba(x_cands), a_min=1e-4, a_max=None)
        # mdre_preds = np.log(preds_prob)
        # dr_mdre = mdre_preds[:, 0] - mdre_preds[:, 1]
        dr_mdre = preds_prob[:, 0] / preds_prob[:, 1]

        candidates = x_cands[np.argmax(dr_mdre)][None, :]
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")

        new_x = unnormalize(torch.from_numpy(candidates), bounds=self.problem.bounds)
        return new_x
