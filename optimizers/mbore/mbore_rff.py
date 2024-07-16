import time

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from botorch.sampling.normal import SobolQMCNormalSampler

from utils.helper import configure_logger
from .scalarizers import HypervoumeContribution
from .classifiers import RFF_MLP


class qUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        beta,
        weights,
        sampler=None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.sampler = sampler
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform()
    def forward(self, X):
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        # switch q and b dimension
        X = X.transpose(0, 1)

        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        scalarized_samples = samples.matmul(self.weights)  # n x b x q
        mean = posterior.mean  # b x q x o
        scalarized_mean = mean.matmul(self.weights)  # b x q
        # ucb_samples = (
        #     scalarized_mean
        #     + math.sqrt(self.beta * math.pi / 2)
        #     * (scalarized_samples - scalarized_mean).abs()
        # )
        # return ucb_samples.mean(dim=1).squeeze()
        scalarized_std = scalarized_samples.std(dim=0)
        return (scalarized_mean + self.beta.sqrt() * scalarized_std).squeeze()


class MBORE_RFF:

    def __init__(
        self,
        problem,
        weight_type='ei',
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
        self.weight_type = weight_type # "ei" or "pi"

        # "HypI", "DomRank", "HypCont", "ParEGO"
        self.scalariser = HypervoumeContribution(reference_point=-problem.ref_point.cpu().numpy())
        self.classifier = None
        self.logger = configure_logger(__name__)
        self.logger.info("MBORE_RFF")
        self.clf = RFF_MLP(
            input_dim=self.problem.dim,
            output_dim=1,
            num_hidden_units=64,
            num_rffs=256,
            num_hidden_layers=4,
            dropout_rate=0.0,
            **self.tkwargs
        )
        self.clf.to(**self.tkwargs)

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

    def fit_model(self, X_obs, y_obs, gamma=1/3, batch_size=256, S=100):
        self.clf.fit(X_obs, y_obs, gamma, batch_size=batch_size, S=S)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        """MBORE assumes minimization"""
        assert batch_size == 1, (f"MBORE only supports a single "
                                 f"batch dimension, but got {batch_size} "
                                 f"batch dimensions.")

        # mbore is for minimization, multiply by -1 to turn it into minimization
        y_obs = - y_obs
        # rescale decision vectors to [0, 1] and declare the associated bounds
        X_obs_norm = normalize(X_obs, self.problem.bounds)
        slb, sub = np.zeros(self.n_dim), np.ones(self.n_dim)
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
        self.fit_model(X_obs_norm, -y_scalarized, gamma=1/3, *args, **kwargs)
        self.logger.info(f"Model fitting takes {time.time() - fit_time:.2f}s")

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
        self.acq_func = qUpperConfidenceBound(
            self.clf,
            beta=9.0,
            weights=torch.tensor([1.0], **self.tkwargs),
            sampler=sampler,
        )

        # generate candidate for maximization: n x q x d
        # x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=self.n_candidates, q=1)
        # with torch.no_grad():
        #     acq_values = self.acq_func(x_cands)
        # candidates = x_cands[torch.argmax(acq_values)] 

        # optimize
        opt_time = time.time()
        candidates, _ = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.standard_bounds,
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples, # used for intialization heuristic
            options={"batch_limit": 6, "maxiter": 200},
            sequential=True,
        )
        self.logger.info(f"Optimizing the acquisition function takes {time.time() - opt_time:.2f}s")

        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
