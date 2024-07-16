import time
from typing import Optional

import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning
)
from botorch.utils.multi_objective.pareto import is_non_dominated
import matplotlib.pyplot as plt

from .trust_region import TurboHParams
from .state import TRBOState
from .gen import TS_select_batch_MORBO

from utils.helper import configure_logger


class MORBO:

    def __init__(
        self,
        problem,
        device="cpu",
        dtype=torch.double,
        n_trust_regions: int = TurboHParams.n_trust_regions,
        max_tr_size: int = TurboHParams.max_tr_size,
        min_tr_size: int = TurboHParams.min_tr_size,
        failure_streak: Optional[int] = None,  # This is better to set automatically
        success_streak: int = TurboHParams.success_streak,
        raw_samples: int = TurboHParams.raw_samples,
        n_restart_points: int = TurboHParams.n_restart_points,
        length_init: float = TurboHParams.length_init,
        length_min: float = TurboHParams.length_min,
        length_max: float = TurboHParams.length_max,
        trim_trace: bool = TurboHParams.trim_trace,
        hypervolume: bool = TurboHParams.hypervolume,
        use_ard: bool = TurboHParams.use_ard,
        verbose: bool = TurboHParams.verbose,
        qmc: bool = TurboHParams.qmc,
        track_history: bool = TurboHParams.track_history,
        sample_subset_d: bool = TurboHParams.sample_subset_d,
        fixed_scalarization: bool = TurboHParams.fixed_scalarization,
        winsor_pct: float = TurboHParams.winsor_pct,
        trunc_normal_perturb: bool = TurboHParams.trunc_normal_perturb,
        switch_strategy_freq: Optional[int] = TurboHParams.switch_strategy_freq,
        tabu_tenure: int = TurboHParams.tabu_tenure,
        decay_restart_length_alpha: float = TurboHParams.decay_restart_length_alpha,
        use_noisy_trbo: bool = TurboHParams.use_noisy_trbo,
        use_simple_rff: bool = TurboHParams.use_simple_rff,
        use_approximate_hv_computations: bool = TurboHParams.use_approximate_hv_computations,
        approximate_hv_alpha: Optional[float] = TurboHParams.approximate_hv_alpha,
        restart_hv_scalarizations: bool = True,
        debug: bool = False,
    ):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}

        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.num_restarts = 10
        self.raw_samples = 512
        self.mc_samples = 128

        self.logger = configure_logger(__name__)
        self.logger.info(f"{__name__}")
        self.debug = debug

         # Automatically set the failure streak if it isn't specified
        failure_streak = max(problem.dim // 3, 10) if failure_streak is None else failure_streak
        self.n_initial_points = 2 * (problem.dim + 1)

        batch_size = 1  # default batch size
        problem_name = problem.__class__.__name__
        if problem_name in ["VehicleSafety", "Penicillin", "CarSideImpact", "BraninCurrin", "GMM", "ZDT2"]:
            max_evals = 200
            min_tr_size = 4
        elif problem_name in ["DTLZ2", "DTLZ3", "Rover"]:
            max_evals = 2000
            # batch_size = 50
            min_tr_size = 200

        # Trust region setup
        self.tr_hparams = TurboHParams(
            length_init=length_init,
            length_min=length_min,
            length_max=length_max,
            batch_size=batch_size,
            success_streak=success_streak,
            failure_streak=failure_streak,
            max_tr_size=max_tr_size,
            min_tr_size=min_tr_size,
            trim_trace=trim_trace,
            n_trust_regions=n_trust_regions,
            verbose=verbose,
            qmc=qmc,
            use_ard=use_ard,
            sample_subset_d=sample_subset_d,
            track_history=track_history,
            fixed_scalarization=fixed_scalarization,
            n_initial_points=self.n_initial_points,
            n_restart_points=n_restart_points,
            raw_samples=raw_samples,
            max_reference_point=problem.ref_point,  # max ref_point
            hypervolume=hypervolume,
            winsor_pct=winsor_pct,
            trunc_normal_perturb=trunc_normal_perturb,
            decay_restart_length_alpha=decay_restart_length_alpha,
            switch_strategy_freq=switch_strategy_freq,
            tabu_tenure=tabu_tenure,
            use_noisy_trbo=use_noisy_trbo,
            use_simple_rff=use_simple_rff,
            use_approximate_hv_computations=use_approximate_hv_computations,
            approximate_hv_alpha=approximate_hv_alpha,
            restart_hv_scalarizations=restart_hv_scalarizations,
        )

        num_objectives = problem.num_objectives
        num_constraints = problem.num_constraints if hasattr(problem, "num_constraints") else 0
        num_outputs = num_objectives + num_constraints
        self.constraints = None
        self.objective = None
        self.trbo_state = TRBOState(
            dim=problem.dim,
            max_evals=max_evals,
            num_outputs=num_outputs,
            num_objectives=num_objectives,
            bounds=problem.bounds,
            tr_hparams=self.tr_hparams,
            constraints=self.constraints,
            objective=self.objective,
        )
        self.if_initialize = True
        self.reintialize_tr = False
        self.init_kwargs = {}

        # For saving outputs
        self.n_evals = []
        self.true_hv = []
        self.pareto_X = []
        self.pareto_Y = []
        self.n_points_in_tr = [[] for _ in range(n_trust_regions)]
        self.n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
        self.n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
        self.tr_sizes = [[] for _ in range(n_trust_regions)]
        self.tr_centers = [[] for _ in range(n_trust_regions)]
        self.tr_restarts = [[] for _ in range(n_trust_regions)]
        self.fit_times = []
        self.gen_times = []
        self.true_ref_point = problem.ref_point

    def initialize(self, X_init, y_init):
        # Create initial points
        self.trbo_state.update(
            X=X_init,
            Y=y_init,
            new_ind=torch.full(
                (X_init.shape[0],), 0, dtype=torch.long, device=X_init.device
            ),
        )
        self.trbo_state.log_restart_points(X=X_init, Y=y_init)

        # Initializing the trust regions. This also initializes the models.
        for i in range(self.tr_hparams.n_trust_regions):
            self.trbo_state.initialize_standard(
                tr_idx=i,
                restart=False,
                switch_strategy=False,
                X_init=X_init,
                Y_init=y_init,
            )
        # Update TRs data across trust regions, if necessary
        self.trbo_state.update_data_across_trs()

        # Set the initial TR indices to -2
        self.trbo_state.TR_index_history.fill_(-2)

        n_points = min(self.n_initial_points, self.trbo_state.max_evals - self.trbo_state.n_evals)
        self.all_tr_indices = [-1] * n_points

    def fit_model(self, X_obs, y_obs, batch_size):
        X_cand = X_obs[-batch_size:]
        Y_cand = y_obs[-batch_size:]

        self.all_tr_indices.extend(self.tr_indices.tolist())
        self.trbo_state.tabu_set.log_iteration()
        # Log TR info
        for i, tr in enumerate(self.trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == self.trbo_state.X_history).all(dim=-1))[0] for x in tr.X]
            )
            tr_inds = self.trbo_state.TR_index_history[inds]
            assert len(tr_inds) == len(tr.X)
            self.n_points_in_tr[i].append(len(tr_inds))
            self.n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            self.n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            self.tr_sizes[i].append(tr.length.item())
            self.tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())

        # Append data to the global history and fit new models
        start_fit = time.time()
        self.trbo_state.update(X=X_cand, Y=Y_cand, new_ind=self.tr_indices)
        self.should_restart_trs = self.trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=self.tr_indices,
            batch_size=self.trbo_state.tr_hparams.batch_size,
            verbose=self.trbo_state.tr_hparams.verbose,
        )
        if self.trbo_state.tr_hparams.verbose:
            self.logger.info(f"Time spent on model fitting: {time.time() - start_fit:.1f} seconds")

    def suggest(self):
        # Getting next suggestions
        start_gen = time.time()
        selection_output = TS_select_batch_MORBO(trbo_state=self.trbo_state)
        if self.trbo_state.tr_hparams.verbose:
            self.logger.info(
                f"Time spent on generating candidates: {time.time() - start_gen:.1f} seconds"
            )

        X_cand = selection_output.X_cand
        self.tr_indices = selection_output.tr_indices
        self.all_tr_indices.extend(self.tr_indices.tolist())
        self.trbo_state.tabu_set.log_iteration()
        return X_cand

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        """
        optimize the MORBO, and returns a new candidate and obervation.
        """
        # normalize training input
        # X_obs_norm = normalize(X_obs, self.problem.bounds)

        if self.if_initialize:
            self.initialize(X_obs, y_obs)
            self.if_initialize = False

            new_x = self.suggest()

            if self.debug:
                self.plot_morbo()
            return new_x

        elif self.reintialize_tr:
            X_center = self.init_kwargs["X_init"]
            Y_center = y_obs[-1]
            for i in range(self.trbo_state.tr_hparams.n_trust_regions):
                if self.should_restart_trs[i]:
                    n_points = min(
                        self.trbo_state.tr_hparams.n_restart_points,
                        self.trbo_state.max_evals - self.trbo_state.n_evals
                    )
                    if n_points <= 0:
                        break  # out of budget
                    if self.trbo_state.tr_hparams.verbose:
                        self.logger.info(f"{self.trbo_state.n_evals}) Restarting trust region {i}")
                    self.trbo_state.TR_index_history[self.trbo_state.TR_index_history == i] = -1
                    if self.trbo_state.tr_hparams.restart_hv_scalarizations:
                        self.init_kwargs["Y_init"] = Y_center
                        self.init_kwargs["X_center"] = X_center
                        self.trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i], dtype=torch.long, device=X_center.device
                            ),
                        )
                        self.trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    self.trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=self.switch_strategy,
                        **self.init_kwargs,
                    )
                    if self.trbo_state.tr_hparams.restart_hv_scalarizations:
                        # we initialized the TR with one data point.
                        # this passes historical information to that new TR
                        self.trbo_state.update_data_across_trs()
                    self.tr_restarts[i].append(
                        self.trbo_state.n_evals.item()
                    )  # Where it restarted
            self.reintialize_tr = False

        # update GP models for each objective
        self.fit_model(X_obs, y_obs, batch_size=batch_size)

        self.switch_strategy = self.trbo_state.check_switch_strategy()
        if self.switch_strategy:
            self.should_restart_trs = [True for _ in self.should_restart_trs]
        if any(self.should_restart_trs):
            for i in range(self.trbo_state.tr_hparams.n_trust_regions):
                if self.should_restart_trs[i]:
                    n_points = min(
                        self.trbo_state.tr_hparams.n_restart_points,
                        self.trbo_state.max_evals - self.trbo_state.n_evals
                    )
                    if n_points <= 0:
                        break  # out of budget
                    if self.trbo_state.tr_hparams.verbose:
                        self.logger.info(f"{self.trbo_state.n_evals}) Restarting trust region {i}")
                    self.trbo_state.TR_index_history[self.trbo_state.TR_index_history == i] = -1
                    self.init_kwargs = {}
                    if self.trbo_state.tr_hparams.restart_hv_scalarizations:
                        # generate new trust region
                        X_center = self.trbo_state.gen_new_restart_design()
                        self.init_kwargs["X_init"] = X_center
                        self.reintialize_tr = True
                        # propose to evaluate a new center
                        return X_center

        if self.trbo_state.tr_hparams.verbose:
            self.logger.info(f"Total refill points: {self.trbo_state.total_refill_points}")

        # update trust regions and global models
        self.trbo_state.update_data_across_trs()

        # Getting next suggestions
        new_x = self.suggest()

        if self.debug:
            self.plot_morbo()

        return new_x

    def plot_morbo(self):
        ref_point = self.problem.ref_point
        n_objs = self.problem.num_objectives
        xs = self.trbo_state.X_history
        ys = self.problem(xs)

        pareto = is_non_dominated(ys)
        bd = DominatedPartitioning(ref_point=ref_point, Y=ys)
        u, l = bd.hypercell_bounds

        fig = plt.figure(figsize=(5*n_objs, 4))
        axs = []
        for i in range(n_objs):
            ax = plt.subplot2grid((1, n_objs + 1), (0, i))
            ax.set_title(f"objective {i + 1}")
            ax.scatter(*xs[:-1, :2].t(), s=10)
            ax.scatter(*xs[-1:, :2].t(), marker="*", color="tab:orange", s=50)
            axs.append(ax)

        ax_pareto = plt.subplot2grid((1, n_objs + 1), (0, n_objs))
        ax_pareto.scatter(*ys[~pareto, :2].t(), s=10, alpha=0.3)
        ax_pareto.scatter(*ys[-1:, :2].t(), marker="*", color="tab:orange", s=50)
        ax_pareto.scatter(*l[:, :2].t(), s=10, color='tab:red', label="pareto")
        ax_pareto.plot(*l[:, :2].t(), color='tab:red')
        ax_pareto.scatter(*ref_point[:2].t(), s=10, color='k', label="ref point")
        ax_pareto.set_xlabel('y1')
        ax_pareto.set_ylabel('y2')
        ax_pareto.set_title("pareto")
 
        plt.tight_layout()
        plt.show()
