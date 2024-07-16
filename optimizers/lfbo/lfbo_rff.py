import math
from collections import OrderedDict

import torch


class RFF_MLP(torch.nn.Module):

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
        layers.append((f"input_layer", torch.nn.Linear(input_dim, num_hidden_units)))
        layers.append((f"input_activation", torch.nn.ELU()))
        layers.append((f"dropout", torch.nn.Dropout(dropout_rate)))
        for i in range(num_hidden_layers):
            layers.append((f"hidden_layer_{i}", torch.nn.Linear(num_hidden_units, num_hidden_units)))
            layers.append((f"activation_{i}", torch.nn.ELU()))
            layers.append((f"dropout_{i}", torch.nn.Dropout(dropout_rate)))
        self.layers = torch.nn.Sequential(OrderedDict(layers))

        # RFF layer
        # m
        self.num_rffs = num_rffs
        # w
        self.register_buffer("weights", torch.randn(size=(self.num_rffs, num_hidden_units)))
        # b
        self.register_buffer("bias", torch.rand(self.num_rffs) * 2 * torch.pi)

        self.blr_layer = torch.nn.Linear(
            num_rffs,
            output_dim,
            bias=False,
        )
        # prior distribution of the weights in the blr layer: N(0, I)
        self.blr_layer.weight.data.normal_(0.0, 1.0)
        # precision matrix is simply the inverse of covariance matrix
        self.intialize_precision_matrix()

    def intialize_precision_matrix(self):
        self.precision_matrix = torch.nn.parameter.Parameter(
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


class LFBO_RFF:

    def __init__(
        self,
        input_dim,
        output_dim=1,
        num_hidden_units=64,
        num_rffs=512,
        num_hidden_layers=4,
        dropout_rate=0.2,
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.clf = RFF_MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            num_rffs=num_rffs,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            **self.tkwargs,
        )
        self.clf.to(**self.tkwargs)

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

    def check_input_dims(self, X):
        # check if X has batch dimension
        if len(X.shape) == 1:
            X = X[..., None, None]
        elif len(X.shape) == 2:
            X = X[:, None, :]
        # X should have: num_obs x n_dim x 1
        return X

    def fit(
        self,
        X,
        y,
        gamma=1/3,
        batch_size=64,
        S=100
    ):
        X = self.check_input_dims(X)

        new_X, _, z, w = self.split_good_bad(
            X.detach(), y.detach(), gamma=gamma
        )

        # add batch dim
        train_tensor = [new_X, z.unsqueeze(1), w.unsqueeze(1)]
        train_dataset = torch.utils.data.TensorDataset(*train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))

        N = len(X)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        optimizer = torch.optim.Adam(self.clf.parameters())
        # update the models
        self.clf.train()
        for epochs in range(E):
            self.clf.intialize_precision_matrix()
            for _, (inputs, targets, weights) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs, _ = self.clf(inputs, targets, weights)
                batch_mle_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, targets, weight=weights
                )
                for param in self.clf.blr_layer.parameters():
                    if param.requires_grad:
                        batch_reg_loss = 0.5 * torch.matmul(param, param.t()).view_as(batch_mle_loss)
                batch_loss = batch_mle_loss + batch_reg_loss / M
                batch_loss.backward()
                optimizer.step()
        self.clf.eval()

        # compute the updated covariance matrix
        self.clf.compute_covariance_matrix()

    def probit_approximation(self, logits_mean, logits_var):
        logits_scale = torch.sqrt(1. + logits_var * torch.pi/8.)
        return logits_mean / logits_scale
    
    def sample_weights(self, num_samples=[100]):
        # Thompson sampling
        w_map = self.clf.blr_layer.weight.data
        w_mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            w_map.flatten(), self.clf.Sigma_N
        )
        return w_mvn.sample(num_samples).unsqueeze(-1)

    def predict(self, X, num_samples=[100]):
        self.clf.eval()
        X = self.check_input_dims(X)

        with torch.no_grad():
            logits_mean, phi_rffs = self.clf(X)

        posterior_weight_samples = self.sample_weights(num_samples)
        posterior_prediction_samples = torch.sigmoid(
            torch.matmul(posterior_weight_samples[:, None, ...].transpose(2, 3), phi_rffs[None, ...].transpose(2, 3))
        )

        return torch.sigmoid(logits_mean), posterior_prediction_samples
    
    def probit_predict(self, X):
        self.clf.eval()
        X = self.check_input_dims(X)

        with torch.no_grad():
            logits_mean, phi_rffs = self.clf(X)
            logits_var = torch.matmul(torch.matmul(phi_rffs, self.clf.Sigma_N), phi_rffs.transpose(2, 1))

        logits_adjusted = self.probit_approximation(logits_mean, logits_var)
        pred_adjusted = torch.sigmoid(logits_adjusted)

        return pred_adjusted
