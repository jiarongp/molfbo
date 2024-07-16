from collections import OrderedDict

import torch


class TaskAgnosticModel(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_units,
        num_layers,
        num_features,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = num_features
        self.task_embedding_size = num_features

        feature_layers = []
        feature_layers.append((f"input_layer", torch.nn.Linear(input_dim, num_hidden_units)))
        feature_layers.append((f"input_activation", torch.nn.ELU()))
        for i in range(num_layers - 2):
            feature_layers.append((f"hidden_layer_{i}", torch.nn.Linear(num_hidden_units, num_hidden_units)))
            feature_layers.append((f"activation_{i}", torch.nn.ELU()))
        feature_layers.append((f"feature_layer", torch.nn.Linear(num_hidden_units, num_features)))
        feature_layers.append((f"feature_activation", torch.nn.ELU()))

        self.feature_layers = torch.nn.Sequential(OrderedDict(feature_layers))
        self.mean_layer = torch.nn.Linear(num_features, output_dim)

        self.register_buffer("noise_mean", torch.tensor(0.0))
        self.register_buffer(
            "task_embedding_noise", torch.tensor(0.1)
        )

    def initialize(self, num_obj, num_tasks, device):
        tensor_kwargs = {'dtype': torch.float32, 'device': device}
        self.initial_obj_embeddings = torch.zeros(num_obj + 1, self.task_embedding_size, **tensor_kwargs)
        self.obj_embedding = torch.nn.Embedding(
            num_obj + 1,
            self.task_embedding_size,
            padding_idx=0,
            _weight=self.initial_obj_embeddings,
        )
        self.initial_task_embeddings = torch.zeros(num_tasks + 1, self.task_embedding_size, **tensor_kwargs)
        self.task_embedding = torch.nn.Embedding(
            num_tasks + 1,
            self.task_embedding_size,
            padding_idx=0,
            _weight=self.initial_task_embeddings,
        )

    def forward(self, x, obj_emb, task_emb):
        # task embedding
        if task_emb.dtype == torch.long:
            task_embedding = self.task_embedding(task_emb)
        else:
            task_embedding = task_emb

        # objective embedding
        if obj_emb.dtype == torch.long: 
            obj_embedding = self.obj_embedding(obj_emb)
        else:
            obj_embedding = obj_emb

        if self.training:
            # add noise to input embedding when training
            task_embedding = task_embedding + torch.distributions.Normal(
                loc=self.noise_mean, scale=self.task_embedding_noise
            ).sample(task_embedding.size())

        x = torch.matmul(obj_embedding[:, None, :].transpose(-1, -2), x[:, None, :]).squeeze()
        # Residual Feedfoward Network
        # first pass through input layer
        features = self.feature_layers[0](x)
        # hidden residual block exclude feature layer, activation and random features
        for i in range(1, len(self.feature_layers) - 2):
            if isinstance(self.feature_layers[i], torch.nn.Linear):
                identity = features
                features = self.feature_layers[i](features)
                features += identity
            else:
                features = self.feature_layers[i](features)
        features = self.feature_layers[-2](features)  # feature layer
        features = self.feature_layers[-1](features)  # activation

        mean = self.mean_layer(features)
        residual = (features * task_embedding).sum(dim=-1, keepdim=True)
        outputs = mean + residual

        return features, outputs
