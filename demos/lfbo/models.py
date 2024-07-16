import math
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import trange


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_units=64,
        num_layers=4,
        dropout=0.1
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.activation = nn.ELU()

        layers = []
        layers.append((f"input_layer", torch.nn.Linear(input_dim, num_units)))
        layers.append((f"activation", self.activation))
        # layers.append((f"dropout", self.dropout))
        for i in range(num_layers - 1):
            layers.append((f"layer_{i}", torch.nn.Linear(num_units, num_units)))
            layers.append((f"activation_{i}", self.activation))
            # layers.append((f"dropout_{i}", self.dropout))
        layers.append((f"output_layer", torch.nn.Linear(num_units, output_dim)))

        self.model = torch.nn.Sequential(OrderedDict(layers))
        
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


class BDRE:

    def __init__(
        self,
        input_dim,
        output_dim,
        device="cpu",
        dtype=torch.float64,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}

        self.clf = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.clf.to(**self.tkwargs)
 
    def fit(self, x, y, weight, batch_size=64, S=100):
        optimizer = torch.optim.Adam(self.clf.parameters())
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

        train_tensors = [x, y]
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        N = len(x)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        losses = []
        for epochs in trange(E):
            for _, (inputs, targets) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs)
                batch_loss = loss_fn(
                    outputs, targets
                )
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())

        # plt.plot(losses)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        self.clf.eval()

    def predict(self, X):
        self.clf.eval()
        with torch.no_grad():
            return self.clf(X)
