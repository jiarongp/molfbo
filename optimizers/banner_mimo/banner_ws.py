from typing import Dict, Optional

import torch
import parameterspace as ps

from .utils import metadata_to_training
from .banner import BaNNER
from .meta_blr import MetaBLR
from .trainer_ws import train


class MetaBLR_WS(MetaBLR):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        inference_dtype=torch.float64,
        **model_config
    ):
        super().__init__(
            input_dim,
            output_dim,
            device,
            inference_dtype,
            **model_config
        )

    def meta_fit(
        self,
        meta_data: Dict,
        num_epochs: int,
        batch_size: int,
        **train_config
    ):
        """Fit model to training data.

        Args:
            meta_data: Dictionary with meta-data.
        """
        X, obj_ids, task_ids, y = metadata_to_training(meta_data)
        self.y_scaler.fit(y)
        y_scaled = self.y_scaler.transform(y)

        train(
            model=self.meta_model,
            train_data=(X, obj_ids, task_ids, y_scaled),
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_data=None,
            device=self.device,
            **train_config
        )


class BaNNER_WS(BaNNER):
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        seed: Optional[int] = None,
        num_samples_acquisition_function: int = 5120,
        **blr_kwargs
    ):
        super().__init__(
            search_space,
            seed,
            num_samples_acquisition_function,
            **blr_kwargs
        )
        # meta-learning blr
        del self.meta_blr
        self.meta_blr = MetaBLR_WS(
            input_dim=blr_kwargs['num_features'],
            output_dim=1,
            **blr_kwargs
        )
