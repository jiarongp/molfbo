import math
from typing import Any, Dict, Optional
from collections import OrderedDict

from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch import Tensor
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal


def ensure_2d(X):
    """Ensures the input (X) is two-dimensional"""
    nd = X.ndim

    if nd == 1:
        X = np.reshape(X, (1, -1))
    elif nd > 2:
        raise ValueError(
            "Input data must be two-dimensional. Given dimensions: " f"{nd:d} {X.shape}"
        )
    return X


class ClassifierBase:
    def __init__(
        self, X_obs: np.ndarray, z_obs: np.ndarray, weights: Optional[np.ndarray]
    ):
        assert X_obs.shape[0] == z_obs.shape[0]

        self.X_obs = X_obs
        self.class_labels = z_obs
        self.weights = weights

    def train(self) -> None:
        """train the classifier on the data given during initialisation"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        take in an array/vector of locations X, shape (n, d) or (d, )
        and return the class-conditional probabilities of class 1.
        """
        raise NotImplementedError


class XGBoostClassifier(ClassifierBase):
    def __init__(
        self,
        X_obs: np.ndarray,
        z_obs: np.ndarray,
        weights: Optional[np.ndarray],
        xgb_settings: Dict[str, Any] = {},
    ) -> None:
        super(XGBoostClassifier, self).__init__(X_obs, z_obs, weights)

        # only import xgboost if we're using it
        from xgboost import XGBClassifier

        # default xgboost settings. taken from the BORE paper supplement (J.2)
        # http://proceedings.mlr.press/v139/tiao21a/tiao21a-supp.pdf
        self.xgb_settings = {
            "n_estimators": 100,  # boosting rounds
            "learning_rate": 0.3,  # eta
            "min_child_weight": 1,
            "max_depth": 6,
            # "min_split_loss": 1,
            # the commands below are to avoid warning messages
            "use_label_encoder": False,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            # xgboost seems to be faster without parallelisation, although this
            # may will likely not be true for all use-cases
            "n_jobs": 1,
        }

        # update the settings with any that are user-specified.
        self.xgb_settings.update(xgb_settings)

        # instantiate the classifier
        self.clf = XGBClassifier(**self.xgb_settings)

    def train(self):
        self.clf.fit(self.X_obs, self.class_labels, sample_weight=self.weights)

    def predict(self, X):
        X = ensure_2d(X)
        _, pi_x = self.clf.predict_proba(X).T  # class 0, class 1
        return pi_x


class Posterior(GPyTorchPosterior):
    def __init__(self, distribution, phi_rffs) -> None:
        super().__init__(distribution=distribution)
        self.phi_rffs = phi_rffs

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        # n x b x q x o
        samples = super().rsample_from_base_samples(sample_shape=sample_shape, base_samples=base_samples)
        # n x q x b x o
        posterior_samples = torch.sigmoid(torch.matmul(self.phi_rffs.unsqueeze(2), samples).transpose(1, 2))
        # n x b x q x o
        return posterior_samples

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        samples = super().rsample(sample_shape=sample_shape, base_samples=base_samples)
        posterior_samples = torch.sigmoid(
            torch.matmul(self.phi_rffs[None, ...], samples)
        )
        return posterior_samples

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        mean = self.distribution.mean
        # multiply the mean by the RFFs
        # add q dimension b x q x d
        # switch q and b: q x b x d
        posterior_mean = torch.matmul(self.phi_rffs, mean.transpose(0, 1))

        # switch back q
        # posterior_mean = posterior_mean.transpose(1, 0).squeeze(1)
        return torch.sigmoid(posterior_mean)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        variance = self.distribution.variance
        if not self._is_mt:
            variance = variance.unsqueeze(-1)
        return variance


class RFF_MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units,
        num_rffs,
        num_hidden_layers,
        dropout_rate=0.2,
        device="cpu:0",
        dtype=torch.double,
    ) -> None:
        super().__init__()
        self.tkwargs = {"dtype": dtype, "device": device}

        layers = []
        layers.append((f"input_layer", nn.Linear(input_dim, num_hidden_units)))
        layers.append((f"input_activation", nn.ELU()))
        layers.append((f"dropout", nn.Dropout(dropout_rate)))
        for i in range(num_hidden_layers):
            layers.append((f"hidden_layer_{i}", nn.Linear(num_hidden_units, num_hidden_units)))
            layers.append((f"activation_{i}", nn.ELU()))
            layers.append((f"dropout_{i}", nn.Dropout(dropout_rate)))
        self.layers = nn.Sequential(OrderedDict(layers))

        # RFF layer
        # m
        self.num_rffs = num_rffs
        # w
        self.register_buffer("weights", torch.randn(size=(self.num_rffs, num_hidden_units)))
        # b
        self.register_buffer("bias", torch.rand(self.num_rffs) * 2 * torch.pi)

        self.blr_layer = nn.Linear(
            num_rffs,
            output_dim,
            bias=False,
        )
        # prior distribution of the weights in the blr layer: N(0, I)
        self.blr_layer.weight.data.normal_(0.0, 1.0)
        # precision matrix is simply the inverse of covariance matrix
        self.intialize_precision_matrix()

    def intialize_precision_matrix(self):
        self.precision_matrix = nn.parameter.Parameter(
            torch.eye(self.num_rffs, **self.tkwargs),
            requires_grad=False
        )
        self.register_parameter('precision_matrix', self.precision_matrix)

    def update_precision_matrix(self, phi_rffs, logits, y, w):
        y_pred = torch.sigmoid(logits)
        # this coefficient only appear when using the LFBO loss, i.e. weighted classification
        coef = (y * w + 1.).view(y_pred.shape)
        precision_matrix_batch_update = torch.sum(
                                            coef * \
                                            y_pred * (1 - y_pred) * \
                                            torch.matmul(phi_rffs.transpose(2, 1), phi_rffs),
                                            axis=0
                                        )
        self.precision_matrix.data += precision_matrix_batch_update

    def compute_covariance_matrix(self):
        # default is lower triangle matrix
        L = torch.linalg.cholesky(self.precision_matrix)
        self.Sigma_N = torch.cholesky_inverse(L)

    def rffs_layer(self, phi):
        # input phi has dimension: num_obs x 1 x num_rffs 
        # phi_rffs = sqrt(2a/m) * cos(wx+b)
        return (
            torch.sqrt(torch.tensor(2.0 / self.weights.shape[0])) *
            torch.cos(torch.matmul(phi, self.weights.t()) + self.bias)
        )

    def forward(self, X, y=None, w=None):
        # Residual Feedfoward Network
        # first pass through input layer
        phi = self.layers[0](X)
        # hidden residual block
        for i in range(1, len(self.layers) - 2):
            if isinstance(self.layers[i], torch.nn.Linear):
                identity = phi
                phi = self.layers[i](phi)
                phi += identity
            else:
                phi = self.layers[i](phi)
        phi = self.layers[-2](phi)  # feature layer
        phi = self.layers[-1](phi)  # activation
        
        # RFF layers
        phi_rffs = self.rffs_layer(phi)
        logits = self.blr_layer(phi_rffs)

        if self.training:
            self.update_precision_matrix(phi_rffs, logits, y, w)
        return logits, phi_rffs

    def check_input_dims(self, X):
        # check if X has batch dimension
        if len(X.shape) == 1:
            X = X[..., None, None]
        elif len(X.shape) == 2:
            X = X[:, None, :]
        # X should have: num_obs x n_dim x 1
        return X
    
    @staticmethod
    def split_good_bad(X, y, gamma, acq_type='ei'):
        tau = torch.quantile(torch.unique(y), q=gamma)
        z = torch.less(y, tau)

        if len(X) > 1 and acq_type == 'ei':
            z_idx = z.squeeze()

            x1, z1 = X[z_idx], z[z_idx].to(X.dtype)
            x0, z0 = X, torch.zeros_like(z).to(X.dtype)

            w1 = (tau - y)[z_idx]
            # sometimes w1 is empty
            w1 = w1 / torch.mean(w1) if len(w1) else w1
            w0 = 1 - z0

            x = torch.concat([x1, x0], axis=0)
            z = torch.concat([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = torch.concat([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / torch.mean(w)

        elif len(X) == 1 or acq_type == 'pi':
            x = X
            w = torch.ones_like(z).to(X.dtype)

        return x, y, z, w
    
    def fit(self, X, y, gamma, batch_size=64, S=100):
        X = self.check_input_dims(X)

        new_X, _, z, w = self.split_good_bad(X.detach(), y.detach(), gamma=gamma)

        # add batch dim
        train_tensor = [new_X, z[..., None, None], w[..., None, None]]
        train_dataset = torch.utils.data.TensorDataset(*train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))

        N = len(X)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        optimizer = torch.optim.AdamW(self.parameters())
        # update the models
        losses = []
        self.train()
        for epochs in range(E):
            self.intialize_precision_matrix()
            for _, (inputs, targets, weights) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs, _ = self(inputs, targets, weights)
                batch_mle_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, targets, weight=weights
                )
                for param in self.blr_layer.parameters():
                    if param.requires_grad:
                        batch_reg_loss = 0.5 * torch.matmul(param, param.t()).view_as(batch_mle_loss)
                batch_loss = batch_mle_loss + batch_reg_loss / M
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_mle_loss.detach().item())
        self.eval()
        # compute the updated covariance matrix
        self.compute_covariance_matrix()

    def posterior(self, X):
        self.eval()  # make sure model is in eval mode
        # X: b x q x d
        if len(X.size()) == 2:
            X = X.unsqueeze(0)

        # with torch.no_grad():
        _, phi_rffs = self(X)

        # 1 x num_rffs
        # batch_shape x event shape
        w_map = self.blr_layer.weight.data
        mvn = MultivariateNormal(
            w_map, self.Sigma_N
        )
        posterior = Posterior(distribution=mvn, phi_rffs=phi_rffs)
        return posterior


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_units=32,
        num_layers=4,
        dropout_rate=0.1,
        dtype=torch.float64,
        device='cpu'
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = nn.ELU()
        self.tkwargs = {"dtype": dtype, "device": device}

        layers = []
        layers.append((f"input_layer", nn.Linear(input_dim, num_units)))
        layers.append((f"activation", self.activation))
        layers.append((f"dropout", nn.Dropout(p=dropout_rate)))
        for i in range(num_layers - 1):
            layers.append((f"layer_{i}", nn.Linear(num_units, num_units)))
            layers.append((f"activation_{i}", self.activation))
            layers.append((f"dropout_{i}", nn.Dropout(p=dropout_rate)))
        layers.append((f"output_layer", nn.Linear(num_units, output_dim)))

        self.model = nn.Sequential(OrderedDict(layers))
        self.model.to(**self.tkwargs)
        
    # def forward(self, x):
    #     logits = self.model(x)
    #     return logits

    def forward(self, x):
        features = self.model[0](x)
        for i in range(1, len(self.model) - 2):
            if isinstance(self.model[i], torch.nn.Linear):
                identity = features
                features = self.model[i](features)
                features += identity
            else:
                features = self.model[i](features)
        logits = self.model[-1](features)
        return logits


class AuxillaryClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_units=32,
        num_layers=4,
        dropout_rate=0.1,
        device="cpu",
        dtype=torch.float64,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = nn.ELU()

        layers = []
        layers.append((f"layer_{0}", torch.nn.Linear(input_dim, num_units)))
        layers.append((f"activation_{0}", self.activation))
        layers.append((f"dropout_{0}", nn.Dropout(p=dropout_rate)))
        for i in range(1, num_layers):
            layers.append((f"layer_{i}", torch.nn.Linear(num_units, num_units)))
            layers.append((f"activation_{i}", self.activation))
            layers.append((f"dropout_{i}", nn.Dropout(p=dropout_rate)))
        layers.append((f"output_layer", torch.nn.Linear(num_units, output_dim)))

        self.model = torch.nn.Sequential(OrderedDict(layers))
        self.model.to(**self.tkwargs)

    def forward(self, x):
        features = self.model[0](x)
        for i in range(1, len(self.model) - 2):
            if isinstance(self.model[i], torch.nn.Linear):
                identity = features
                features = self.model[i](features)
                features += identity
            else:
                features = self.model[i](features)
        logits = self.model[-1](features)
        return logits


    def fit(self, X, bounds, batch_size=64, S=500):
        optimizer = torch.optim.AdamW(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        x_p = X
        N = 2*len(x_p)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.model.train()
        losses = []
        for _ in trange(E):
            x_m = draw_sobol_samples(bounds=bounds, n=len(x_p), q=1).squeeze(1)
            x = torch.concat([x_p, x_m], axis=0)
            z_p= torch.empty(x_p.shape[0], dtype=torch.long).fill_(0)
            z_m = torch.empty(x_m.shape[0], dtype=torch.long).fill_(1)
            z = torch.concat([z_p, z_m], axis=0)
            labels = torch.nn.functional.one_hot(z).to(x.dtype)

            s_p = x_p.shape[0]
            s_m = x_m.shape[0]
            w_p = torch.tensor(s_p * [(s_p + s_m) / s_m]).to(x.dtype)
            w_m = torch.tensor(s_m * [(s_p + s_m) / s_m]).to(x.dtype)
            w = torch.cat([w_p, w_m], axis=0)
            w = w / w.mean()

            train_dataset = torch.utils.data.TensorDataset(x, labels, w)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            for _, (inputs, labels, weights) in enumerate(train_dataloader):
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                batch_loss = (weights * loss).mean()
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())

        self.model.eval()
