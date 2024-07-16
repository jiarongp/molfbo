import math
import time
from typing import Optional

import torch
from torch import nn
import numpy as np
from sklearn import preprocessing
from botorch.utils.transforms import unnormalize, normalize

from utils.helper import configure_logger
from .scalarizers import HypervoumeContribution
from .optimizers import logistic_lbfgs_optimizer


def mse_loss(input, target, weight):
    return (weight * (input - target)**2).sum()

def loss_fn(intermediate_output, output, target_y, target_class, weight):
    mse_losses = torch.zeros(1, 1)
    for i in range(target_y.shape[-1]): 
        mse_losses += mse_loss(intermediate_output[:, i], target_y[:, i], weight=weight)
    nll_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target_class, weight)
    return nll_loss + mse_losses


class COMPOSITE_MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        obj_dim,
        obj_output_dim,
        final_output_dim,
        num_units=32,
        dtype=torch.float64,
        device='cpu',
    ):
        super().__init__()
        self.obj_dim = obj_dim
        self.obj_output_dim = obj_output_dim
        self.intermediate_dim = obj_dim * obj_output_dim
        self.tkwargs = {"dtype": dtype, "device": device}

        self.model_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, obj_output_dim),
            ) for _ in range(obj_dim)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(self.intermediate_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, final_output_dim),
            nn.ReLU()
        )
        self.to(**self.tkwargs)

    def forward(self, x):
        intermidiate_preds = []
        for i, _ in enumerate(self.model_list):
            intermidiate_pred = self.model_list[i](x)
            intermidiate_preds.append(intermidiate_pred)
        intermidiate_preds = torch.cat(intermidiate_preds, dim=1)
        logits = self.classifier(intermidiate_preds)
        return intermidiate_preds, logits

    def fit(self, x, y, z, w, batch_size, S, debug):
        self.y_scaler = preprocessing.StandardScaler().fit(y)
        y_scaled = self.y_scaler.transform(y)

        x, y, z, w = (
            torch.tensor(x, **self.tkwargs),
            torch.tensor(y_scaled, **self.tkwargs),
            torch.tensor(z, **self.tkwargs),
            torch.tensor(w, **self.tkwargs)
        )

        optimizer = torch.optim.AdamW(self.parameters())

        train_tensors = [x, y, z.unsqueeze(-1), w.unsqueeze(-1)]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        N = len(x)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.train()
        losses = []
        for epochs in range(E):
            for _, (inputs, targets_y, targets_class, weights) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                intermediate_outputs, outputs = self(inputs)
                batch_loss = loss_fn(intermediate_outputs, outputs, targets_y, targets_class, weights)
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())

        if debug:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3, 3))
            plt.plot(losses)
            plt.title("Training loss over iterations")
        self.eval()


class MBORE_COMPOSITE:

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
        # budget for optimising model (acq func equivalent)# budget for optimising model (acq func equivalent)
        self.optimization_budget = 1024 * problem.dim
        self.acq_type = acq_type # "ei" or "pi"
        # "HypI", "DomRank", "HypCont", "ParEGO"
        self.scalariser = HypervoumeContribution(reference_point=-problem.ref_point.cpu().numpy())

        self.debug = debug
        self.logger = configure_logger(__name__)
        self.logger.info(f"{__name__}")

    @staticmethod
    def load_classification_data(
        X: np.ndarray,
        y: np.ndarray,
        y_original: np.ndarray,
        gamma: float,
        acq_type: Optional[str],
        dupe_points: bool = True,
    ):
        # https://github.com/lfbo-ml/lfbo/blob/7d3364dd0eeab5ac2cfcfcb17c02473d89146195/model.py
        tau = np.quantile(y, q=gamma)
        z = np.less(y, tau)
        print(f"tau: {-tau}")

        # avoid error when there are only negative samples
        acq_type = "pi" if z.sum() == 0 else acq_type

        if acq_type == "ei":
            # split the points into two classes
            x1, y1, y1_original, z1 = X[z], y[z], y_original[z], z[z]

            if dupe_points:
                x0, y0, y0_original, z0 = X, y, y_original, np.zeros_like(z)
            else:
                x0, y0, y0_original, z0 = X[~z], y[~z], y_original, np.zeros(np.count_nonzero(~z), dtype="bool")

            w1 = (tau - y)[z]
            w1 = w1 / np.mean(w1)
            w0 = 1 - z0

            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w1 = w1 * (s1 + s0) / s1
            w0 = w0 * (s1 + s0) / s0

            x = np.concatenate([x1, x0], axis=0)
            y = np.concatenate([y1, y0], axis=0)
            y_original = np.concatenate([y1_original, y0_original], axis=0)
            z = np.concatenate([z1, z0], axis=0)

            w = np.concatenate([w1, w0], axis=0)
            w = w / np.mean(w)

            return x, y, y_original, z, w, tau

        elif (acq_type == "pi") or (acq_type is None):
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            w = np.ones_like(z)

            return X, y, y_original, z, w, tau

        else:
            raise ValueError(f"weight_type must be 'ei' or 'pi'/None, given: {acq_type}")

    @staticmethod
    def objective_thresholding(y_obs, ref_point, offset=1e-9):
        if len(ref_point.shape) == 1:
            ref_point = ref_point.reshape(1, -1)
        y_threshold = np.repeat(ref_point - offset, y_obs.shape[0], axis=0)
        better_than_ref = (y_obs < ref_point).all(axis=-1)
        y_threshold[better_than_ref] = y_obs[better_than_ref]
        return y_threshold

    def fit_model(self, X_obs, y_obs, y_scalarized, gamma, batch_size=64, S=500):
        x, y, y_original, z, w, tau = self.load_classification_data(
            X_obs, y_scalarized, y_obs, gamma=gamma, acq_type=self.acq_type
        )

        self.clf = COMPOSITE_MLP(
            input_dim=self.problem.dim,
            obj_dim=self.problem.num_objectives,
            obj_output_dim=1,
            final_output_dim=1,
            **self.tkwargs
        )
        self.clf.fit(x, y_original, z, w, batch_size, S, debug=self.debug)
 

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, gamma=1/3, batch_size=1, *args, **kwargs):
        """MBORE assumes minimization"""
        assert batch_size == 1, (f"MBORE only supports a single "
                                 f"batch dimension, but got {batch_size} "
                                 f"batch dimensions.")

        # mbore is for minimization, multiply by -1 to turn it into minimization
        y_obs_neg = -y_obs.detach().cpu().numpy()

        # rescale decision vectors to [0, 1] and declare the associated bounds
        X_obs_norm = normalize(X_obs, self.problem.bounds).detach().cpu().numpy()

        # NOTE: objective thresholding, this is a modification due to the ref_point setting in BoTorch
        y_obs_thresholded = self.objective_thresholding(y_obs_neg, ref_point=-self.problem.ref_point.numpy())

        # scalarise the objective values
        _, hypi = self.scalariser.get_ranks(y_obs_thresholded, return_scalers=True)

        # negate the scalarized since we want to maximize the hypervolume contribution
        # therefore minimize its negative
        fit_time = time.time()
        self.fit_model(X_obs_norm, y_obs, -hypi, gamma, *args, **kwargs)
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        opt_time = time.time()
        # perform the optimisation of the predictive distribution. note that this point
        # is in the scaled space (i.e. [0, 1]^d)
        candidate = logistic_lbfgs_optimizer(self.clf, self.standard_bounds, self.optimization_budget)
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")
        candidate = torch.tensor(candidate, **self.tkwargs).reshape(1, -1)
        new_x = unnormalize(candidate, bounds=self.problem.bounds)

        # debug plots
        if self.debug:
            self.plot_2d(
                torch.from_numpy(X_obs_norm)[:, :2],
                y_obs[:, :2],
                candidate
            )
            # self.plot_hypi(y_scalarized, gamma)

        return new_x

    def plot_2d(self, X_obs, y_obs, candidate):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

        n_obj = self.problem.num_objectives
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
            intermediate_preds, pred_logits = self.clf(x_cands)
        pred_y = self.clf.y_scaler.inverse_transform(intermediate_preds)
        score = torch.nn.functional.sigmoid(pred_logits)
        y_cands = self.problem(x_cands)[:, :2]
        new_x = unnormalize(candidate, bounds=self.problem.bounds)
        y_candidate = self.problem(new_x)[:, :2]

        fig = plt.figure(figsize=(5 * (n_obj+3), 5))

        # plot input space
        axs = []
        for i in range(n_obj):
            ax = plt.subplot2grid((1, n_obj+3), (0, i))
            # plot functions
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
            countourset = ax.contourf(xx, yy, pred_y[:, i].reshape(xx.shape),)
            ax.scatter(*X_obs.t(), alpha=0.3, s=10, color='tab:red')
            ax.scatter(*candidate.t(), s=100, marker="*", color="tab:orange")
            cax.grid(False)  # just to remove the warning
            clb = fig.colorbar(countourset, cax=cax)
            clb.ax.set_title('z')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f"objective {i + 1}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            axs.append(ax)

        ax_pred = plt.subplot2grid((1, n_obj+3), (0, 2))
        # plot functions
        countourset = ax_pred.contourf(xx, yy, score.reshape(xx.shape),)
        ax_pred.scatter(*X_obs.t(), alpha=0.3, s=10, color='tab:red')
        ax_pred.scatter(*candidate.t(), s=100, marker="*", color="tab:orange")
        ax_pred.set_xlabel('x1')
        ax_pred.set_ylabel('x2')
        ax_pred.set_title(f"Score")
        ax_pred.set_xlim(0, 1)
        ax_pred.set_ylim(0, 1)

        # plot the predictive
        ax_mo_pred = plt.subplot2grid((1, n_obj + 3), (0, n_obj+1))
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
        ax_pareto = plt.subplot2grid((1, n_obj+3), (0, n_obj+2))
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
        tau = np.quantile(-y_obs_scalarized, q=gamma)
        z = np.less(-y_scalarized, tau)
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
        ax_gamma.set_title("input space HyPI with gamma")

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
        ax_pareto_gamma.set_title('outspace HyPI with gamma')
        plt.tight_layout()
        plt.show()
