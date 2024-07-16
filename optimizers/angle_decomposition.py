import time
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

from utils.helper import configure_logger


class MLP(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        device="cpu:0",
        dtype=torch.double,
    ) -> None:
        super().__init__()
        self.tkwargs = {"device": device, "dtype": dtype}

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def compute_ideal_pt(y_obs):
    ideal_pt, _ = y_obs.max(dim=0)
    return ideal_pt


def compute_nadir_pt(y_obs):
    nadir, _ = y_obs.min(dim=0)
    return nadir


def y_normalize(y_obs, nadir_pt, ideal_pt):
    return (y_obs - nadir_pt) / (ideal_pt - nadir_pt)


class AngleDecomp:

    def __init__(
        self,
        problem,
        gamma=1/3,
        device="cpu:0",
        dtype=torch.double,
        weight_type='ei',
        debug=True,
    ) -> None:
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1

        # budget for optimising model (acq func equivalent)# budget for optimising model (acq func equivalent)
        self.n_dim = problem.dim
        self.n_obj = problem.num_objectives
        self.n_candidates = min(5000, max(2000, 200 * problem.dim))
        self.gamma = gamma # proportion of solutions to include in the *good* class
        self.weight_type = weight_type # "ei" or "pi"
        
        self.debug = debug
        self.logger = configure_logger(__name__)
        self.logger.info("AngleDecomp")

    @staticmethod
    def load_classification_data(
        X: np.ndarray,
        y: np.ndarray,
        gamma: float,
        weight_type: Optional[str],
        dupe_points: bool = True,
    ):
        # https://github.com/lfbo-ml/lfbo/blob/7d3364dd0eeab5ac2cfcfcb17c02473d89146195/model.py
        tau = np.quantile(np.unique(y), q=gamma)
        z = np.less(y, tau)

        # avoid error when there are only negative samples
        weight_type = "pi" if z.sum() == 0 else weight_type

        if weight_type == "ei":
            # split the points into two classes
            x1, y1, z1 = X[z], y[z], z[z]

            if dupe_points:
                x0, y0, z0 = X, y, np.zeros_like(z)
            else:
                x0, y0, z0 = X[~z], y[~z], np.zeros(np.count_nonzero(~z), dtype="bool")

            w1 = (tau - y)[z]
            w1 = w1 / np.mean(w1)
            w0 = 1 - z0

            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w1 = w1 * (s1 + s0) / s1
            w0 = w0 * (s1 + s0) / s0

            x = np.concatenate([x1, x0], axis=0)
            y = np.concatenate([y1, y0], axis=0)
            z = np.concatenate([z1, z0], axis=0)

            w = np.concatenate([w1, w0], axis=0)
            w = w / np.mean(w)

            return x, y, z, w, tau

        elif (weight_type == "pi") or (weight_type is None):
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            w = np.ones_like(z)

            return X, y, z, w, tau

        else:
            raise ValueError(f"weight_type must be 'ei' or 'pi'/None, given: {weight_type}")
        
    def generate_angle_data(self, X_obs, y_obs, rho_obs, tau):
        z = torch.less(torch.from_numpy(-rho_obs), tau)
        x_interest = X_obs[z]
        y_interest = y_obs[z]

        ref_dirs = get_reference_directions("das-dennis", n_dim=self.problem.dim, n_partitions=8)
        ref_dirs = torch.tensor(ref_dirs, **self.tkwargs)
        # ref_dirs = ref_dirs* (self.ideal_pt - self.nadir_pt) + self.nadir_pt

        proj_len = torch.matmul(y_interest, ref_dirs.t())
        l2_norm = torch.linalg.norm(ref_dirs, ord=2, dim=-1, keepdim=True)
        proj_len /= l2_norm.t()
        max_len = proj_len.max(dim=1)

        angle_count = torch.zeros_like(proj_len)
        for i in range(proj_len.shape[0]):
            max_len = proj_len[i].max()
            angle_count[i] = (proj_len[i] == max_len)

        x_angle = x_interest
        y_angle = torch.matmul(angle_count, angle_count.sum(dim=0))
        return x_angle, y_angle, ref_dirs

    def fit_model(self, X_obs, y_obs):
        rho_obs = np.linalg.norm(y_obs, ord=2, axis=1)
        x_l_train, y_l_train, z_l_train, w_l_train, tau = self.load_classification_data(
            X_obs, -rho_obs, gamma=1/3, weight_type=self.weight_type
        )
        if self.debug:
            self.plot_length(X_obs, rho_obs, tau)

        x_angle, y_angle, ref_dirs = self.generate_angle_data(
            X_obs, y_obs, rho_obs, tau
        )
        x_a_train, y_a_train, z_a_train, w_a_train, tau = self.load_classification_data(
            x_angle, y_angle.numpy(), gamma=1/3, weight_type=self.weight_type
        )

        if self.debug:
            self.plot_angle(x_a_train, z_a_train, ref_dirs)

        x_l_train, z_l_train, w_l_train = (
            torch.tensor(x_l_train, **self.tkwargs),
            torch.tensor(z_l_train, **self.tkwargs).unsqueeze(1),
            torch.tensor(w_l_train, **self.tkwargs).unsqueeze(1)
        )
        x_a_train, z_a_train, w_a_train = (
            torch.tensor(x_a_train, **self.tkwargs),
            torch.tensor(z_a_train, **self.tkwargs).unsqueeze(1),
            torch.tensor(w_a_train, **self.tkwargs).unsqueeze(1)
        ) 

        self.length_model = MLP(self.n_dim, 1, **self.tkwargs)
        self.length_model.to(**self.tkwargs)
        self.length_model = self.train_model(x_l_train, z_l_train, w_l_train, self.length_model)

        self.angle_model = MLP(self.n_dim, 1, **self.tkwargs)
        self.angle_model.to(**self.tkwargs)
        self.angle_model = self.train_model(x_a_train, z_a_train, w_a_train, self.angle_model)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        X_obs_norm = normalize(X_obs, self.problem.bounds)

        self.nadir_pt = compute_nadir_pt(y_obs)
        self.ideal_pt = compute_ideal_pt(y_obs) 
        y_obs = y_normalize(y_obs, self.nadir_pt, self.ideal_pt)

        fit_time = time.time()
        self.fit_model(X_obs_norm, y_obs)
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        # generate candidate for maximization: n x q x d
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=self.n_candidates, q=1).squeeze(1)
        with torch.no_grad():
            angle_scores = self.angle_model(x_cands)
            angle_scores = torch.sigmoid(angle_scores)
            length_scores = self.length_model(x_cands)
            length_scores = torch.sigmoid(length_scores)
        score = angle_scores * length_scores 

        candidates = x_cands[torch.argmax(score)].unsqueeze(0)
        if self.debug:
            self.plot_aggregate(x_cands, angle_scores, length_scores, score, candidates)
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x

    def train_model(self, x_train, z_train, w_train, model):
        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        # add batch dimension
        train_tensors = [x_train, z_train.to(**self.tkwargs), w_train]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        self.length_model.train()
        for epochs in range(1000):
            for _, (inputs, targets, weights) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                batch_loss = loss_fn(outputs, targets, weight=weights)
                batch_loss.backward()
                optimizer.step()
        model.eval()
        return model

    def plot_aggregate(self, x, angle_scores, length_scores, agg_scores, candidates):
        for n, score in enumerate([angle_scores, length_scores, agg_scores]):
            fig = plt.figure(figsize=(9, 3))
            axs = []
            for i in range(self.n_obj):
                ax = plt.subplot2grid((1, self.n_obj + 1), (0, i))
                ax.scatter(*x.t(), s=1, cmap='bwr', c=score)
                ax.scatter(*candidates.t(), s=50, marker="*", color="tab:orange")
                ax.set_title(f"objective {i + 1}")
                axs.append(ax)

            ax_pareto = plt.subplot2grid((1, self.n_obj + 1), (0, self.n_obj))
            ax_pareto.set_xlabel('y1')
            ax_pareto.set_ylabel('y2')
            ax_pareto.set_title("pareto")

            y = self.problem(x)
            ax_pareto.scatter(*y.t(), s=1, cmap='bwr', c=score)
            ax_pareto.scatter(*self.problem(candidates).t(), s=50, marker="*", color="tab:orange")
            # ax_pareto.legend()
            plt.tight_layout()
            plt.show()

    def plot_length(self, x_obs, rho_obs, tau):
        z = torch.less(torch.from_numpy(-rho_obs), tau)

        fig = plt.figure(figsize=(9, 3))
        axs = []
        for i in range(self.n_obj):
            ax = plt.subplot2grid((1, self.n_obj + 1), (0, i))
            ax.set_title(f"objective {i + 1}")
            axs.append(ax)

        ax_pareto = plt.subplot2grid((1, self.n_obj + 1), (0, self.n_obj))
        ax_pareto.set_xlabel('y1')
        ax_pareto.set_ylabel('y2')
        ax_pareto.set_title("pareto")

        good_obs = x_obs[z.squeeze()]
        bad_obs = x_obs[~z.squeeze()]
        ax_pareto.scatter(*self.problem(good_obs).t(), s=10, color='tab:red')
        ax_pareto.scatter(*self.problem(bad_obs).t(), s=10, color='tab:blue')

        axs[0].scatter(*bad_obs.t(), s=10, color="tab:blue", alpha=.8)
        axs[0].scatter(*good_obs.t(), s=10, color="tab:red", alpha=.8)
        axs[1].scatter(*bad_obs.t(), s=10, color="tab:blue", alpha=.8)
        axs[1].scatter(*good_obs.t(), s=10, color="tab:red", alpha=.8)
        plt.tight_layout()
        plt.show()

    def plot_angle(self, x_train, z_train, ref_dirs):
        import matplotlib as mpl
        cmap=mpl.colormaps["tab10"]
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(9, 3))
        axs = []
        for i in range(self.n_obj):
            ax = plt.subplot2grid((1, self.n_obj + 1), (0, i))
            ax.set_title(f"objective {i + 1}")
            axs.append(ax)

        ax_pareto = plt.subplot2grid((1, self.n_obj + 1), (0, self.n_obj))
        ax_pareto.set_xlabel('y1')
        ax_pareto.set_ylabel('y2')
        ax_pareto.set_title("pareto")

        ref_dirs = ref_dirs * (self.ideal_pt - self.nadir_pt) + self.nadir_pt
        for n, v in enumerate(ref_dirs):
            line = Line2D(
                [self.nadir_pt[0], v[0]],
                [self.nadir_pt[1], v[1]],
                linewidth=1, linestyle = "-",
                color=cmap(n),
                alpha=0.8,
            )
            ax_pareto.add_line(line)

        x_train = torch.tensor(x_train, **self.tkwargs)
        good_obs = x_train[z_train.squeeze()]
        bad_obs = x_train[~z_train.squeeze()]
        ax_pareto.scatter(*self.problem(bad_obs).t(), s=10, color='tab:blue')
        ax_pareto.scatter(*self.problem(good_obs).t(), s=10, color='tab:red')

        axs[0].scatter(*bad_obs.t(), s=10, color="tab:blue", alpha=.8)
        axs[0].scatter(*good_obs.t(), s=10, color="tab:red", alpha=.8)
        axs[1].scatter(*bad_obs.t(), s=10, color="tab:blue", alpha=.8)
        axs[1].scatter(*good_obs.t(), s=10, color="tab:red", alpha=.8)

        plt.tight_layout()
        plt.show()
