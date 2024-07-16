import torch
import torch.nn.utils.spectral_norm as sn


class Generator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, output_dim),
            torch.nn.Tanh()
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            sn(torch.nn.Linear(input_dim, 128)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            sn(torch.nn.Linear(128, 128)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            sn(torch.nn.Linear(128, 128)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            sn(torch.nn.Linear(128, 1)),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
