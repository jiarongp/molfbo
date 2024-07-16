import math
import torch


class JointMLP(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        device="cpu:0",
        dtype=torch.double,
    ) -> None:
        super().__init__()
        self.tkwargs = {"device": device, "dtype": dtype}

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def forward(self, x, gamma):
        input = torch.concat([x, gamma], dim=-1)
        logits = self.layers(input)
        return logits


class LFBO_JointRand:

    def __init__(
        self,
        input_dim,
        output_dim,
        weight_type='ei',
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.clf = JointMLP(
            input_dim=input_dim + 1, # plus 1 for gamma
            output_dim=output_dim,
            device=device,
            dtype=dtype,
        )
        self.clf.to(**self.tkwargs)
        self.weight_type = weight_type

    @staticmethod
    def split_good_bad(X, y, gamma, weight_type):
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

    def fit(self, X_obs, y_obs, batch_size=64, S=1000):

        optimizer = torch.optim.AdamW(self.clf.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        N = len(X_obs)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        for epochs in range(E):
            gamma = torch.rand(1, **self.tkwargs)
            new_X, _, new_z, new_w = self.split_good_bad(
                X_obs.detach(), y_obs.detach(), gamma=gamma, weight_type=self.weight_type
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
                optimizer.zero_grad()
                outputs = self.clf(inputs, quantile)
                batch_loss = loss_fn(outputs.squeeze(1), targets, weight=weights)
                batch_loss.backward()
                optimizer.step()
        self.clf.eval()

    def predict(self, X, gamma):
        quantile = torch.tensor([gamma] * X.shape[0], **self.tkwargs)

        # add batch dimension
        if len(quantile.shape) == 1:
            quantile = quantile[..., None, None]

        with torch.no_grad():
            return torch.sigmoid(self.clf(X.unsqueeze(1), quantile)).squeeze(-2)
