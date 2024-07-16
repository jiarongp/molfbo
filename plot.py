import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from botorch.utils.multi_objective.pareto import is_non_dominated

from utils.helper import load_json


def plot_hvs(
    data,
    problem_description,
    n_batch,
    batch_size,
    save_dir,
    plot_regret=False,
    log_scale=False,
):
    iters = np.arange(n_batch + 1) * batch_size

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for opt, result in data.items():

        opt_hvs = np.asarray(result["hvs"])
        hv_mean = np.asarray(opt_hvs).mean(axis=0)
        if plot_regret and problem_description["max_hv"] is not None:
            hv_mean = problem_description["max_hv"] - hv_mean
        hv_ci = opt_hvs.std(axis=0) / np.sqrt(opt_hvs.shape[0])

        if log_scale:
            ax.plot(iters, np.log10(hv_mean), label=opt)
            ax.fill_between(
                iters,
                np.log10(hv_mean + 2 * hv_ci),
                np.log10(np.clip(hv_mean - 2 * hv_ci, 1e-3, None)),
                alpha=.1
            )
        else:
            ax.plot(iters, hv_mean, label=opt)
            ax.fill_between(
                iters,
                hv_mean + 2 * hv_ci,
                hv_mean - 2 * hv_ci,
                alpha=.1
            )
    # ax.set_ylim(-1, None)
    ax.set_xlim(0, None)
    ax.grid('on')
    ax.set_title(problem_description["name"])
    ax.set(
        xlabel="number of observations (beyond initial points)",
        ylabel="Log Hypervolume Difference" if plot_regret else "Hypervolume",
    )
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    fig.savefig(save_dir / "hvs_difference.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_cost(
    data,
    problem_description,
    save_dir,
    plot_regret=False
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for opt, result in data.items():
        traj_df = pd.DataFrame()
        for hvs, cost in zip(result["hvs"], result['cost']):
            cost = np.cumsum(cost)
            traj_df = traj_df.join(
                # index is the interpolated running cost
                # num of columns means number of runs
                pd.DataFrame({len(traj_df.columns): hvs}, index=cost),
                how='outer'
            )
        traj_df = traj_df.ffill()
        traj_df = traj_df.bfill()

        running_cost = np.array(traj_df.index)
        opt_hvs = np.array(traj_df.T)

        hv_mean = opt_hvs.mean(axis=0)
        if plot_regret and problem_description["max_hv"] is not None:
            hv_mean = problem_description["max_hv"] - hv_mean
        hv_ci = opt_hvs.std(axis=0) / np.sqrt(opt_hvs.shape[0])

        ax.plot(running_cost, np.log10(hv_mean), label=opt)
        ax.fill_between(
            running_cost,
            np.log10(hv_mean + 2 * hv_ci),
            np.log10(np.clip(hv_mean - 2 * hv_ci, 1e-9, None)),
            alpha=.1
        )

    ax.grid('on')
    ax.set_title(problem_description["name"])
    ax.set(
        xlabel="Cost (beyond initial cost)",
        ylabel="Log Hypervolume Difference" if plot_regret else "Log Hypervolume",
    )
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(data.keys()))
    fig.savefig(save_dir / "hvs_cost.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_pareto(
    data,
    problem_description,
    n_batch,
    batch_size,
    save_dir,
    run_idx=0,
    plot_ref_point=True
):
    # only plot the one of the run for visualization
    num_opt = len(data)

    fig = plt.figure(figsize=(5 * num_opt + 3, 7))
    cm = plt.colormaps["viridis"]

    batch_number = np.concatenate(
        [
            np.zeros(2 * (problem_dim + 1)),
            np.arange(1, n_batch + 1).repeat(batch_size),
        ]
    )

    for i, (opt, result) in enumerate(data.items()):
        y_obs = torch.tensor(result["y"][run_idx])
        ax = fig.add_subplot(1, num_opt, i+1, sharex=ax if i > 0 else None, sharey=ax if i > 0 else None)
        ax.scatter(
            y_obs[:, 0],
            y_obs[:, 1],
            c=batch_number,
            alpha=0.8,
            label="observations"
        )
        if plot_ref_point:
            ax.scatter(*problem_description["ref_point"], marker='x', color='tab:red', label="reference point")
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            ax.axhspan(y_lim[0], problem_description["ref_point"][1], alpha=0.2, color='tab:gray')
            ax.axvspan(x_lim[0], problem_description["ref_point"][0], alpha=0.2, color='tab:gray')
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)

        pareto = is_non_dominated(y_obs)
        sort_idx = np.argsort(y_obs[pareto][:, 0])
        plt.plot(*y_obs[pareto][sort_idx].T, label='approximated pareto front', color='tab:red')

        ax.set_title(opt)
        ax.set_xlabel("Objective 1")

        if i == 0:
            ax.set_ylabel("Objective 2")
            ax.legend()

    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title("Iteration")

    fig.suptitle(problem_description["name"])
    plt.tight_layout()
    plt.savefig(save_dir / "pareto_front.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmarks.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True
    )
    parser.add_argument(
        "--plot_decoupled",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--plot_pareto",
        action="store_false"
    )
    args = parser.parse_args()

    for benchmark in args.benchmarks:

        results_dir = Path(args.root) / benchmark
        data = dict()
        for rs in results_dir.glob("*.json"):
            opt_run_dict = load_json(results_dir, filename=rs.name)
            for k, v in opt_run_dict.items():
                data[k] = v

        problem_description = data.pop("problem_description")
        seed = data.pop("seed")
        optimizers = list(data.keys())

        # compute batch size and iterations
        x_example = np.asarray(data[optimizers[0]]["x"])
        hvs_example = np.asarray(data[optimizers[0]]["hvs"])
        problem_dim = x_example.shape[-1]
        n_initial = 2 * (problem_dim + 1)
        n_batch = hvs_example.shape[-1] - 1  # minus the initial points
        n_obs = x_example.shape[-2] - n_initial
        batch_size = int(n_obs / n_batch)

        if args.plot_decoupled:
            plot_cost(data, problem_description, results_dir)
        else:
            plot_hvs(data, problem_description, n_batch, batch_size, results_dir)
            if args.plot_pareto:
                # plot_pareto(data, problem_description, n_batch, batch_size, results_dir)
                pass
