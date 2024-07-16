import math
from collections import OrderedDict

import torch


class JointMLP(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units,
        num_layers,
        device="cpu:0",
        dtype=torch.double,
    ) -> None:
        super().__init__()
        self.tkwargs = {"device": device, "dtype": dtype}

        # define shared features
        layers = []
        layers.append((f"input_layer", torch.nn.Linear(input_dim, num_hidden_units)))
        layers.append((f"input_activation", torch.nn.ELU()))
        for i in range(num_layers):
            layers.append((f"hidden_layer_{i}", torch.nn.Linear(num_hidden_units, num_hidden_units)))
            layers.append((f"activation_{i}", torch.nn.ELU()))
        layers.append((f"output_layer", torch.nn.Linear(num_hidden_units, output_dim)))

        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x, gamma):
        """
        x: input
        """
        input = torch.concat([x, gamma], dim=-1)
        logits = self.layers(input)
        return logits


class LFBO_Joint:

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units=32,
        num_layers=4,
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.clf = JointMLP(
            input_dim=input_dim + 1, # plus 1 for gamma
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        self.clf.to(**self.tkwargs)

    @staticmethod
    def split_good_bad(X, y, gamma):
        tau = torch.quantile(torch.unique(y), q=gamma)
        z = torch.less(y, tau)

        if len(X) > 1:
            z_idx = z.squeeze()

            x1, z1 = X[z_idx], z[z_idx]
            x0, z0 = X, torch.zeros_like(z)

            w1 = (tau - y)[z_idx]
            # sometimes w1 is empty
            w1 = w1 / torch.mean(w1) if len(w1) else w1
            w0 = 1 - z0.int()

            x = torch.concat([x1, x0], axis=0)
            z = torch.concat([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = torch.concat([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / torch.mean(w)

        elif len(X) == 1:
            x = X
            w = torch.ones_like(z)

        return x, y, z, w 

    def fit(
        self,
        X_obs,
        y_obs,
        num_gammas=20,
        batch_size=64,
        S=100
    ):
        optimizer = torch.optim.Adam(self.clf.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        gammas = torch.rand(num_gammas, 1, **self.tkwargs)
        X = torch.empty(0, X_obs.shape[-1], **self.tkwargs)
        z = torch.empty(0, **self.tkwargs)
        w = torch.empty(0, **self.tkwargs)
        quantile = torch.empty(0, 1, **self.tkwargs)

        for gamma in gammas:
            new_X, _, new_z, new_w = self.split_good_bad(
                X_obs.detach(), y_obs.detach(), gamma=gamma
            )
            new_quantile = torch.tensor([gamma] * len(new_X))

            X = torch.concat([X, new_X], dim=0)
            z = torch.concat([z, new_z], dim=0)
            w = torch.concat([w, new_w], dim=0)
            quantile = torch.concat([quantile, new_quantile.unsqueeze(-1)], dim=0)

        # add batch dimension
        train_tensors = [X[..., None], z, w, quantile[..., None]]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        N = len(X_obs)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        for epochs in range(E):
            for _, (inputs, targets, weights, quantile) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs, quantile)
                batch_loss = loss_fn(
                    outputs.squeeze(), targets, weight=weights
                )
                batch_loss.backward()
                optimizer.step()
        self.clf.eval()

    def predict(self, X, gamma):
        quantile = torch.tensor([gamma] * X.shape[0], **self.tkwargs)

        # add batch dimension
        if len(X.shape) == 2:
            X = X.unsqueeze(-2) 
        if len(quantile.shape) == 1:
            quantile = quantile[..., None, None]

        with torch.no_grad():
            return torch.sigmoid(self.clf(X, quantile)).squeeze()