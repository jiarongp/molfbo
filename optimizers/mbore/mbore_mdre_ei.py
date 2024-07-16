import math
import time

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

from utils.helper import configure_logger
from .scalarizers import HypervoumeContribution
from .classifiers import MLP
from .optimizers import lbfgs_optimizer


class MBORE_MDRE_EI:

    def __init__(
        self,
        problem,
        acq_type='ei',
        device="cpu",
        dtype=torch.double,
        debug=False,
    ) -> None:
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        # budget for optimising model (acq func equivalent)# budget for optimising model (acq func equivalent)
        self.optimization_budget = 1024 * problem.dim
        self.acq_type = acq_type # "ei" or "pi"
        # "HypI", "DomRank", "HypCont", "ParEGO"
        self.scalariser = HypervoumeContribution(reference_point=-problem.ref_point.cpu().numpy())

        self.debug = debug
        self.logger = configure_logger(__name__)
        self.logger.info(f"{__name__}")

    @staticmethod
    def objective_thresholding(y_obs, ref_point, offset=1e-9):
        if len(ref_point.shape) == 1:
            ref_point = ref_point.reshape(1, -1)
        y_threshold = np.repeat(ref_point - offset, y_obs.shape[0], axis=0)
        better_than_ref = (y_obs < ref_point).all(axis=-1)
        y_threshold[better_than_ref] = y_obs[better_than_ref]
        return y_threshold

    def generate_mdre_dataset(self, X, y, gamma, acq_type='ei'):
        tau = torch.quantile(y, q=gamma)
        z = torch.less(y, tau)
        self.logger.info(f"tau: {-tau}")

        # avoid error when there are only negative samples
        acq_type = "pi" if z.sum() == 0 else acq_type

        if len(X) > 1 and acq_type == 'ei':
            x_p, y_p = X[z], y[z]
            w_p = (tau - y)[z]
            w_p = w_p / torch.mean(w_p)

            x_q, y_q = X, y
            x_m = draw_sobol_samples(bounds=self.standard_bounds, n=len(x_p), q=1).squeeze(1)

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
            w_m = torch.tensor(len(z_m) * [ (s_q + s_p + s_m) / s_m]).to(X.dtype)
            w = torch.cat([w_p, w_q, w_m], axis=0)
            w = w / w.mean()

            return x, z_onehot, w 

        elif len(X) == 1 or acq_type == 'pi':
            x_p, y_p = X[z], y[z]
            x_q, y_q = X[~z], y[~z]
            # squeeze the q dimension 
            x_m = draw_sobol_samples(bounds=self.standard_bounds, n=len(x_q), q=1).squeeze(1)
            self.x_m = x_m
            x = torch.cat([x_p, x_q, x_m], axis=0)
            
            z_p = torch.empty(x_p.shape[0], dtype=torch.long).fill_(0)
            z_q = torch.empty(x_q.shape[0], dtype=torch.long).fill_(1)
            z_m = torch.empty(x_m.shape[0], dtype=torch.long).fill_(2)
            z = torch.cat([z_p, z_q, z_m], axis=0)
            z_onehot = torch.nn.functional.one_hot(z).to(X.dtype)

            w = torch.ones(len(x), 1, dtype=X.dtype)
            
            return x, z_onehot, w

    def fit_model(self, X_obs, y_obs, gamma, batch_size=64, S=500):
        x, z, w = self.generate_mdre_dataset(
            X_obs.detach(), y_obs.detach(), gamma=gamma, acq_type=self.acq_type
        )
        self.clf = MLP(
            input_dim=self.problem.dim,
            output_dim=3,
            num_units=64,
            num_layers=4,
            dropout_rate=0.1,
            **self.tkwargs
        )
        optimizer = torch.optim.AdamW(self.clf.parameters())
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        train_tensors = [x, z, w]  # normalize the weights
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
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs)
                loss = loss_fn(
                    outputs, targets
                )
                batch_loss = (loss * weights).mean()
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())
        self.clf.eval()

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, gamma=1/3, batch_size=1, *args, **kwargs):
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
        opt_time = time.time()
        # perform the optimisation of the predictive distribution. note that this point
        # is in the scaled space (i.e. [0, 1]^d)
        candidate = lbfgs_optimizer(self.clf, self.standard_bounds, self.optimization_budget)
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")
        candidate = torch.tensor(candidate, **self.tkwargs).reshape(1, -1)
        new_x = unnormalize(candidate, bounds=self.problem.bounds)

        # debug plots
        if self.debug:
            self.plot_2d(
                X_obs_norm[:, :2],
                torch.from_numpy(-y_obs)[:, :2],
                candidate
            )
            self.plot_hypi(y_scalarized, gamma)

        return new_x

    def plot_2d(self, X_obs, y_obs, candidate):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

        xx, yy = np.meshgrid(
            np.linspace(
                *self.problem.bounds.t()[0], 100
            ),
            np.linspace(
                *self.problem.bounds.t()[1], 100
            ),
        )
        x_cands = np.vstack((xx.flatten(), yy.flatten())).T
        x_cands = torch.from_numpy(x_cands).to(**self.tkwargs)
        with torch.no_grad():
            pred_logits = self.clf(x_cands)
        pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
        score = pred_probs[:, 0] / pred_probs[:, 1]
        y_cands = self.problem(x_cands)[:, :2]
        new_x = unnormalize(candidate, bounds=self.problem.bounds)
        y_candidate = self.problem(new_x)[:, :2]

        fig = plt.figure(figsize=(5 * 4, 9))
        # plot input space
        axs = []
        for i, pred_name in enumerate(['p', 'q', 'm', "aggregate"]):
            ax = plt.subplot2grid((2, 4), (0, i))
            # plot functions
            if i < 3:
                countourset = ax.contourf(xx, yy, pred_probs[:, i].reshape(xx.shape),)
            else:
                countourset = ax.contourf(xx, yy, score.reshape(xx.shape),)
            # ax.tricontourf(*X_obs[:, :2].t(), scores)
            ax.scatter(*X_obs.t(), alpha=0.3, s=10, color='tab:red')
            ax.scatter(*self.x_m.t(), s=10, color="tab:green")
            ax.scatter(*candidate.t(), s=100, marker="*", color="tab:orange")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f"{pred_name}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            axs.append(ax)

        # plot the predictive
        ax_mo_pred = plt.subplot2grid((2, 4), (1, 0))
        div = make_axes_locatable(ax_mo_pred)
        cax = div.append_axes('right', '5%', '5%')
        ax_mo_pred.scatter(*y_cands.t(), s=5, cmap="viridis", c=score, alpha=0.5)
        ax_mo_pred.scatter(*y_candidate.t(), s=100, marker="*", color="tab:orange")
        ax_mo_pred.scatter(*self.problem.ref_point.t(), s=10, color='k', label="ref point")

        cax.grid(False)  # just to remove the warning
        clb = fig.colorbar(countourset, cax=cax)
        clb.ax.set_title('z')
        ax_mo_pred.set_xlabel('y1')
        ax_mo_pred.set_ylabel('y2')
        ax_mo_pred.set_title('pareto prediction')

        # plot observations
        ax_pareto = plt.subplot2grid((2, 4), (1, 1))
        pareto = is_non_dominated(y_obs)
        bd = DominatedPartitioning(ref_point=self.problem.ref_point, Y=y_obs)
        u, l = bd.hypercell_bounds
        ax_pareto.scatter(*y_obs[~pareto].t(), s=10, alpha=0.3)
        ax_pareto.scatter(*l.t(), s=10, color='tab:red')
        ax_pareto.plot(*l.t(), color='tab:red', label="pareto")
        ax_pareto.scatter(*y_candidate.t(), s=100, marker="*", color="tab:orange")
        ax_pareto.scatter(*self.problem.ref_point.t(), s=10, color='k', label="ref point")
        ax_pareto.set_xlabel('y1')
        ax_pareto.set_ylabel('y2')
        ax_pareto.set_title('pareto')
        ax_pareto.legend()

        plt.tight_layout()
        plt.show()

    def plot_hypi(self, y_obs_scalarized, gamma):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        xx, yy = np.meshgrid(
            np.linspace(
                *self.problem.bounds.t()[0], 100
            ),
            np.linspace(
                *self.problem.bounds.t()[1], 100
            ),
        )
        x_cands = np.vstack((xx.flatten(), yy.flatten())).T
        x_cands = torch.from_numpy(x_cands).to(**self.tkwargs)
        y_cands = self.problem(x_cands)[:, :2]
        y_cands_filtered = self.objective_thresholding(-y_cands.numpy(), ref_point=-self.problem.ref_point.numpy())
        # scalarise the objective values
        _, y_scalarized = self.scalariser.get_ranks(y_cands_filtered, return_scalers=True)
        tau = np.quantile(y_obs_scalarized, q=gamma)
        z = np.greater(y_scalarized, tau)
        y_scalarized_gamma = np.zeros_like(y_scalarized)
        y_scalarized_gamma[z] = y_scalarized[z]

        fig = plt.figure(figsize=(16, 4))
        # plot input space
        ax = plt.subplot2grid((1, 4), (0, 0))
        # plot functions
        countourset = ax.contourf(xx, yy, y_scalarized.reshape(xx.shape))
        # ax.tricontourf(*X_obs[:, :2].t(), scores)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("input space HyPI")

        # plot the predictive
        ax_pareto = plt.subplot2grid((1, 4), (0, 1))
        div = make_axes_locatable(ax_pareto)
        cax = div.append_axes('right', '5%', '5%')
        ax_pareto.scatter(*y_cands.T, s=5, cmap="viridis", c=y_scalarized)
        cax.grid(False)  # just to remove the warning
        clb = fig.colorbar(countourset, cax=cax)
        clb.ax.set_title('z')
        ax_pareto.set_xlabel('y1')
        ax_pareto.set_ylabel('y2')
        ax_pareto.set_title('outspace HyPI')

        ax_gamma = plt.subplot2grid((1, 4), (0, 2))
        # plot functions
        countourset = ax_gamma.contourf(xx, yy, y_scalarized_gamma.reshape(xx.shape))
        ax_gamma.set_xlabel('x1')
        ax_gamma.set_ylabel('x2')
        ax_gamma.set_xlim(0, 1)
        ax_gamma.set_ylim(0, 1)
        ax_gamma.set_title("input space HyPI")

        # plot the predictive
        ax_pareto_gamma = plt.subplot2grid((1, 4), (0, 3))
        div = make_axes_locatable(ax_pareto_gamma)
        cax = div.append_axes('right', '5%', '5%')
        ax_pareto_gamma.scatter(*y_cands.T, s=5, cmap="viridis", c=y_scalarized_gamma)
        cax.grid(False)  # just to remove the warning
        clb = fig.colorbar(countourset, cax=cax)
        clb.ax.set_title('z')
        ax_pareto_gamma.set_xlabel('y1')
        ax_pareto_gamma.set_ylabel('y2')
        ax_pareto_gamma.set_title('outspace HyPI')
        plt.tight_layout()
        plt.show()
