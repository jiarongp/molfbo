from typing import Optional

import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)


class TchebyshevThompsonSampling(MaxPosteriorSampling):
    def __init__(
            self,
            model: Model,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            replacement: bool = True,
        ) -> None:
        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement
        )

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False,
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        # remove num_samples dim
        return self.maximize_samples(X, samples, num_samples).squeeze(1)
