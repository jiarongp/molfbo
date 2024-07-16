import math
from collections import OrderedDict

import torch


class MLP(torch.nn.Module):

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

        layers = []
        layers.append((f"input_layer", torch.nn.Linear(input_dim, num_hidden_units)))
        layers.append((f"input_activation", torch.nn.ELU()))
        for i in range(num_layers):
            layers.append((f"hidden_layer_{i}", torch.nn.Linear(num_hidden_units, num_hidden_units)))
            layers.append((f"activation_{i}", torch.nn.ELU()))
        layers.append((f"output_layer", torch.nn.Linear(num_hidden_units, output_dim)))

        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, X):
        logits = self.layers(X)
        return logits


class LFBO_MLP:

    def __init__(
        self,
        input_dim,
        output_dim=1,
        num_hidden_units=32,
        num_layers=4,
        weight_type='ei',
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.clf = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            num_layers=num_layers,
            device=device,
            dtype=dtype
        )
        self.clf.to(**self.tkwargs)
        self.weight_type = weight_type

    @staticmethod
    def split_good_bad(X, y, gamma, weight_type='ei'):
        tau = torch.quantile(torch.unique(y), q=gamma)
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
        X,
        y,
        gamma=0.33,
        batch_size=64,
        S=100
    ):
        optimizer = torch.optim.Adam(self.clf.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        new_X, _, z, w = self.split_good_bad(
            X.detach(), y.detach(), gamma=gamma, weight_type=self.weight_type
        )

        # assuming the shape of x is: num_examples x num_dim
        # adding batch dimension
        train_tensors = [new_X.unsqueeze(1), z.to(**self.tkwargs), w]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        N = len(X)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        for epochs in range(E):
            for _, (inputs, targets, weights) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs)
                batch_loss = loss_fn(
                    outputs.squeeze(1), targets, weight=weights
                )
                batch_loss.backward()
                optimizer.step()
        self.clf.eval()

    def predict(self, X):
        with torch.no_grad():
            # adding batch dimension
            return torch.sigmoid(self.clf(X.unsqueeze(1)))
