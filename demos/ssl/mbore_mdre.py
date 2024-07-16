import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

from utils.helper import configure_logger
from optimizers.mbore.scalarizers import HypervoumeContribution
from optimizers.mbore.classifiers import MLP, AuxillaryClassifier


class MBORE_MDRE_EI_AUX:

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

    def generate_mdre_dataset(self, X, y, gamma, acq_type='ei'):
        # self.auxi_clf = AuxillaryClassifier(input_dim=self.problem.dim, output_dim=2, **self.tkwargs)
        # self.auxi_clf.fit(X, self.standard_bounds, batch_size=256, S=500)

        tau = torch.quantile(y, q=gamma)
        z = torch.less(y, tau)

        if len(X) > 1 and acq_type == 'ei':
            x_p, y_p = X[z], y[z]
            w_p = (tau - y)[z]
            w_p = w_p / torch.mean(w_p)

            x_q, y_q = X, y

            x_m = draw_sobol_samples(bounds=self.standard_bounds, n=len(X), q=1).squeeze(1)
            self.x_m = x_m
            x = torch.cat([x_p, x_q, x_m], axis=0)

            z_p = torch.empty(x_p.shape[0], dtype=torch.long).fill_(0)
            z_q = torch.empty(x_q.shape[0], dtype=torch.long).fill_(1)
            z_m = torch.empty(x_m.shape[0], dtype=torch.long).fill_(2)
            z = torch.cat([z_p, z_q, z_m], axis=0)
            z_onehot = torch.nn.functional.one_hot(z).to(X.dtype)

            s_p = x_p.shape[0]
            s_q = x_q.shape[0]
            s_m = x_m.shape[0]

            w_p = w_p * (s_p + s_q + s_m) / s_p
            w_q = z_q * (s_p + s_q + s_m) / s_q
            w_m = z_q * (s_p + s_q + s_m) / s_m
            w = torch.cat([w_p, w_q, w_m], axis=0)
            w = w / w.mean()
            # with torch.no_grad():
            #     logits = self.auxi_clf(x)
            # importance_ratio = (logits[:, 1] - logits[:, 0]).exp()
            # w = importance_ratio * w
       
            return x, z_onehot, w 

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

    def fit_model(self, X_obs, y_obs, gamma, batch_size=256, S=500):
        x, z, w = self.generate_mdre_dataset(
            X_obs.detach(), y_obs.detach(), gamma=gamma, acq_type=self.acq_type
        )
        self.clf = MLP(
            input_dim=self.problem.dim,
            output_dim=3,
            num_units=64,
            num_layers=4,
            dropout_rate=0.0,
            **self.tkwargs
        )
        optimizer = torch.optim.AdamW(self.clf.parameters())
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        train_tensors = [x, z, w]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        N = len(x)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        losses = []
        for epochs in range(E):
            for _, (inputs, targets, weights) in enumerate(train_dataloader):
                optimizer.zero_grad()

                outputs = self.clf(inputs)
                batch_loss = loss_fn(
                    outputs, targets
                )
                # batch_loss = (batch_loss * weights).mean()
                batch_loss = (batch_loss * weights / weights.sum()).sum()
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())

        self.clf.eval()

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, gamma=1/3, total_steps=200, *args, **kwargs):
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
        X_obs_norm, y_scalarized = torch.from_numpy(X_obs_norm).to(**self.tkwargs), torch.from_numpy(y_scalarized).to(**self.tkwargs)

        fit_time = time.time()
        self.fit_model(X_obs_norm, -y_scalarized, gamma=gamma, *args, **kwargs)
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        # generate candidate for maximization: n x q x d
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=self.n_candidates, q=1).squeeze(1)
        opt_time = time.time()
        with torch.no_grad():
            pred_logits = self.clf(x_cands)
        dr_mdre = pred_logits[:, 0] - pred_logits[:, 1]

        candidates = x_cands[torch.argmax(dr_mdre)].unsqueeze(0)
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x

    def estimate_kl_divergence(self):
        with torch.no_grad():
            log_prob = self.auxi_clf(draw_sobol_samples(bounds=self.standard_bounds, n=1024, q=1).squeeze(1))
        log_ratio = log_prob[:, 1] - log_prob[:, 0] 
        kl = log_ratio.mean()
        return kl