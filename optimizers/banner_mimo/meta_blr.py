from typing import Dict

import torch
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import StandardScaler

from .trainer import train
from .meta_model import TaskAgnosticModel
from .utils import metadata_to_training
from .inference import marginalized_blr_nll, marginalized_blr_predict


class MetaBLR:
    """
    Meta-learning Bayesian Linear Regression

    This classifier contains two components:
        1. A task-agnostic model (g_w), which meta-learns on the meta-data
        and fixed during inference. It provides the features representation
        shared across task and mean predicition for warm-starting optimization. 
        2. A task-specific layer parameterized by z, which is a Bayesian
        logistic regression using the meta-learned features shared across
        task. It performs adaptation on the target task.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        inference_dtype=torch.float64,
        **model_config
    ):
        super().__init__()
        self.input_dim = input_dim,
        self.output_dim = output_dim
        self.inference_dtype = inference_dtype
        self.model_config = model_config

        # assign device automatically
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.beta = 1e3
        # Task-agnostic meta-learning
        self.meta_model = TaskAgnosticModel(
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config
        ).to(self.device)
        self.y_scaler = StandardScaler()
        self.hist_phi = torch.empty(
            0, self.meta_model.num_features, dtype=self.inference_dtype
        )
        self.hist_y = torch.empty(0, 1, dtype=self.inference_dtype)

    @property
    def to(self, device: str):
        self.meta_model.to(device)

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

    def fit(self, X, y, **kwargs):
        """Fit the task-specific Bayesian logistic regression on target task data"""

        self.meta_model.eval().to(device=self.device, dtype=self.inference_dtype)
        X = torch.tensor(X, device=self.device, dtype=self.inference_dtype)
        # TODO: need to normalize y before inference
        y_scaled = self.y_scaler.transform(y)
        y_scaled = torch.tensor(y_scaled, device=self.device, dtype=self.inference_dtype).reshape(-1, self.output_dim)

        self._fit(X, y_scaled, **kwargs)

    def _fit(self, X, y):
        """Fit the task-specific Bayesian logistic regression on target task data"""

        embedding = torch.tensor(0, device=self.device)
        with torch.no_grad():
            features, mean_outputs = self.meta_model(X, embedding)

        self.hist_phi = features
        self.y_res = y - mean_outputs

        # Optimize hyperparameter beta
        def min_me(log_beta):
            log_beta = torch.as_tensor(
                log_beta, dtype=self.inference_dtype, device=self.device
            )
            nll = marginalized_blr_nll(
                self.hist_phi,
                self.hist_y,
                # setting alpha to 1.0, meaning fixing the prior Gaussian to
                # a standard Gaussian. This is the same as adding the regularization
                # term in the paper.
                alpha=torch.tensor(1.0, device=self.device),
                beta=torch.exp(log_beta),
            )
            return nll.detach().cpu().numpy()

        with torch.no_grad():
            # using global optimization to find the best
            # beta, that minimize the nll
            res = opt.differential_evolution(
                min_me,
                [(np.log(1), np.log(1e6))],
                maxiter=100,
            )

        self.beta = torch.exp(res.x[0]).item()

    def predict(self, X, *args, **kwargs):
        X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            mean, var = self._predict(X, *args, **kwargs)

        mean = mean.detach().cpu().numpy()
        var = var.detach().cpu().numpy()

        # inverse transform the output to original scale
        mean = self.y_scaler.inverse_transform(mean)
        var = (self.y_scaler.scale_ ** 2) * var

        return mean, var

    def _predict(self, X, obj_embedding, task_embedding=None):
        self.meta_model.eval().to(dtype=self.inference_dtype)

        if task_embedding is None:
            task_embedding = torch.tensor(0, device=self.device)
        with torch.no_grad():
            features, mean = self.meta_model(X, obj_embedding, task_embedding)

        res_mean, var = marginalized_blr_predict(
            phi=features,
            hist_phi=self.hist_phi,
            hist_y=self.hist_y,
            alpha=1.0,
            beta=self.beta,
        )

        return mean + res_mean, var
