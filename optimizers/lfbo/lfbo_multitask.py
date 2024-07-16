import math
from collections import OrderedDict

import torch


class MultitaskMLP(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units,
        num_features,
        num_layers,
        num_tasks,
        device="cpu:0",
        dtype=torch.double,
    ) -> None:
        super().__init__()
        self.tkwargs = {"device": device, "dtype": dtype}
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.task_embedding_size = num_features

        # define shared features
        feature_layers = []
        feature_layers.append((f"input_layer", torch.nn.Linear(input_dim, num_hidden_units)))
        feature_layers.append((f"input_activation", torch.nn.ELU()))
        for i in range(num_layers - 2):
            feature_layers.append((f"hidden_layer_{i}", torch.nn.Linear(num_hidden_units, num_hidden_units)))
            feature_layers.append((f"activation_{i}", torch.nn.ELU()))
        feature_layers.append((f"feature_layer", torch.nn.Linear(num_hidden_units, num_features)))
        feature_layers.append((f"feature_activation", torch.nn.ELU()))

        self.feature_layers = torch.nn.Sequential(OrderedDict(feature_layers))

        # define tasks layer
        initial_weight_embeddings = torch.zeros(
            num_tasks,
            self.task_embedding_size,
            **self.tkwargs
        )
        self.task_embedding = torch.nn.Embedding(
            num_tasks,
            self.task_embedding_size,
            _weight=initial_weight_embeddings,
        )

    def forward(self, x, quantile_id):
        """
        x: input
        task_id: the index decide which task layer to predict, start with 0
        """
        features = self.feature_layers(x)
        weight_embedding = self.task_embedding(quantile_id)
        logits = torch.einsum('ijk,ijk->i', features, weight_embedding)
        return logits


class LFBO_MultiTask:

    def __init__(
        self,
        input_dim,
        output_dim,
        gammas,
        num_hidden_units=32,
        num_features=32,
        num_layers=4,
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.gammas = gammas
        self.clf = MultitaskMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            num_features=num_features,
            num_layers=num_layers,
            num_tasks=len(gammas),
            **self.tkwargs
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
        batch_size=64,
        S=100
    ):
        optimizer = torch.optim.Adam(self.clf.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        X = torch.empty(0, X_obs.shape[-1], **self.tkwargs)
        z = torch.empty(0, **self.tkwargs)
        w = torch.empty(0, **self.tkwargs)
        quantile_ids = torch.empty(0, dtype=torch.long, device=self.tkwargs["device"])
        for quantile, gamma in enumerate(self.gammas):
            new_X, _, new_z, new_w = self.split_good_bad(
                X_obs.detach(), y_obs.detach(), gamma=gamma
            )
            new_quantile_id = torch.tensor([quantile] * len(new_X))

            X = torch.concat([X, new_X], dim=0)
            z = torch.concat([z, new_z], dim=0)
            w = torch.concat([w, new_w], dim=0)
            quantile_ids = torch.concat([quantile_ids, new_quantile_id], dim=0)

        # add batch dimension
        train_tensors = [X[..., None], z, w, quantile_ids[..., None]]
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
            for _, (inputs, targets, weights, quantile_id) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs, quantile_id)
                batch_loss = loss_fn(
                    outputs, targets, weight=weights
                )
                batch_loss.backward()
                optimizer.step()
        self.clf.eval()

    def predict(self, X, quantile_id):
        quantile_id = torch.tensor(quantile_id, dtype=torch.long)

        # add batch dimension
        if len(quantile_id.shape) == 1:
            quantile_id = quantile_id[..., None]
        if len(X.shape) == 2:
            X = X.unsqueeze(-2)

        with torch.no_grad():
            return torch.sigmoid(self.clf(X, quantile_id))