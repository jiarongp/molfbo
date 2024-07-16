import time
import argparse
import logging
import warnings
from pathlib import Path

import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from utils.helper import save_json
from optimizers import *
from benchmarks import *

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}


def generate_initial_data(problem, noise_std=0., n=6):
    # generate training data
    noise = torch.ones(problem.num_objectives, **tkwargs) * noise_std
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise
    # negative values imply feasibility in botorch
    train_con = -problem.evaluate_slack(train_x)
    return train_x, train_obj, train_con


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run benchmarks.")
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARKS.keys(),
        required=True
    )
    parser.add_argument(
        "--optimizers",
        choices=OPTIMIZERS.keys(),
        required=True,
        nargs="+"
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        nargs='?',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs='?',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--studies",
        type=int,
        required=True
    )
    parser.add_argument(
        "--output",
        type=str
    )
    parser.add_argument(
        "--verbose",
        action="store_false"
    )
    args = parser.parse_args()

    # check argument evaluations and budgets
    if not(hasattr(args, "evaluations") or hasattr(args, "budgets")):
        raise ValueError("Must specify arguments evaluations, budgets or both")

    logger = logging.getLogger(f"{args.benchmark}_benchmark")
    logger.info(f"Device: {tkwargs['device'].type}\ndtype: {tkwargs['dtype']}")
    output_dir = Path("results") if args.output is None else Path(args.output)


    for study in range(1, args.studies + 1):
        # define problem
        problem = BENCHMARKS[args.benchmark](**BENCHMARKS_KWS[args.benchmark], negate=True).to(**tkwargs)

        # define noise
        noise_std = 0.
        noise = torch.ones(problem.num_objectives, **tkwargs) * noise_std

        # save problem description once for all benchmark
        if study == 1:
            try:
                max_hv = problem.max_hv
            except NotImplementedError:
                max_hv = None

            problem_description = {
                "problem_description":{
                    "name": BENCHMARKS[args.benchmark].__name__,
                    "max_hv": max_hv,
                    "ref_point": problem.ref_point.tolist()
                }
            }
            save_json(problem_description, output_dir=output_dir, filename=f"problem_description.json")

        init_x, init_y, init_con = generate_initial_data(problem, noise_std=noise_std, n=2 * (problem.dim + 1))

        for opt in args.optimizers:
            run_results = dict()
            # optimizer
            optimizer = OPTIMIZERS[opt](problem, **tkwargs)
            X_obs, y_obs, con_obs = init_x, init_y, init_con

            # compute hypervolume:
            # NOTE: DominatedPartitioning always assumes maximization, i.e. the ref_point should smaller all possible (interested) objective.
            # Internally, it multiplies outcomes by -1 and performs the decomposition under minimization.
            # NOTE: DominatedPartitioning does objective thresholding, meaning all the point that are out of the range defined by the ref_point
            # are set to be zero hypervolume. Example is shown: 
            # https://ax.dev/tutorials/multiobjective_optimization.html#Set-Objective-Thresholds-to-focus-candidate-generation-in-a-region-of-interest
            hvs = []

            is_feas = (con_obs <= 0).all(dim=-1)
            y_feas = y_obs[is_feas]
            if y_feas.shape[0] > 0:
                # compute feasible hypervolume
                bd = DominatedPartitioning(ref_point=problem.ref_point, Y=y_feas)
                volume = bd.compute_hypervolume().item()
            else:
                volume = 0.0
            hvs.append(volume)

            # run N_ITER rounds of BayesOpt after the initial random batch
            N_ITER = args.evaluations if hasattr(args, "evaluations") else torch.inf

            iteration = 1
            while (iteration < N_ITER + 1):
                t0 = time.monotonic()

                # propose new candidate
                new_x = optimizer.observe_and_suggest(X_obs, y_obs, batch_size=1)

                # evaluate new candidate
                new_y = problem(new_x)
                # negative values imply feasibility in botorch
                new_con = -problem.evaluate_slack(new_x)

                # observe new candidate
                X_obs = torch.cat([X_obs, new_x])
                y_obs = torch.cat([y_obs, new_y])
                con_obs = torch.cat([con_obs, new_con])

                # update progress
                is_feas = (con_obs <= 0).all(dim=-1)
                y_feas = y_obs[is_feas]
                if y_feas.shape[0] > 0:
                    # compute feasible hypervolume
                    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=y_feas)
                    volume = bd.compute_hypervolume().item()
                else:
                    volume = 0.0
                hvs.extend([volume]*len(new_x))

                # NSGA evaluates a batch of points
                iteration += len(new_x)
                t1 = time.monotonic()
                if args.verbose:
                    logger.info(
                        f"\n{opt} batch {iteration:>2}: Hypervolume = "
                        f"({hvs[-1]:>4.2f}), "
                        f"time = {t1-t0:>4.2f}.",
                        end="",
                    )
                else:
                    logger.info(".", end="")

            run_results[opt] = {}
            run_results[opt]["hvs"] = hvs
            run_results[opt]["x"] = X_obs.detach().cpu().tolist()
            run_results[opt]["y"] = y_obs.detach().cpu().tolist()

            save_json(run_results, output_dir=output_dir, filename=f"{opt}_{study:02}.json")