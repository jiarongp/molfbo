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


class LFBO_JointRand:

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units=64,
        num_layers=4,
        weight_type='ei',
        interpolation='lower',
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
        self.weight_type = weight_type
        self.interpolation = interpolation

    @staticmethod
    def split_good_bad(X, y, gamma, weight_type, interpolation):
        tau = torch.quantile(torch.unique(y), q=gamma, interpolation=interpolation)
        z = torch.less(y, tau)

        if len(X) > 1 and weight_type == 'ei':
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

        elif len(X) == 1 or weight_type == 'pi':
            x = X
            w = torch.ones_like(z)

        return x, y, z, w 

    def fit(
        self,
        X_obs,
        y_obs,
        batch_size=64,
        S=100
    ):
        optimizer = torch.optim.Adam(self.clf.parameters())
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        N = len(X_obs)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        for epochs in range(E):
            gamma = torch.rand(1, **self.tkwargs)
            new_X, _, new_z, new_w = self.split_good_bad(
                X_obs.detach(), y_obs.detach(), gamma=gamma, weight_type=self.weight_type, interpolation=self.interpolation
            )
            new_quantile = torch.tensor([[gamma]] * len(new_X), **self.tkwargs)

            # add batch dimension
            train_tensors = [new_X.unsqueeze(1), new_z.to(**self.tkwargs), new_w, new_quantile.unsqueeze(1)]
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            for _, (inputs, targets, weights, quantile) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs, quantile)
                batch_loss = loss_fn(
                    outputs.squeeze(1), targets, weight=weights
                )
                batch_loss.backward()
                optimizer.step()
            # scheduler.step()
        self.clf.eval()

    def predict(self, X, gamma):
        quantile = torch.tensor([gamma] * X.shape[0], **self.tkwargs)

        # add batch dimension
        if len(quantile.shape) == 1:
            quantile = quantile[..., None, None]

        with torch.no_grad():
            return torch.sigmoid(self.clf(X.unsqueeze(1), quantile)).squeeze(-2)