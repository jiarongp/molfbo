import torch

from models import Generator, Discriminator
from training import gan_training


if __name__ == "__main__":

    mix = torch.distributions.Categorical(torch.ones(2,))
    comp = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.vstack((5 * torch.ones(2,), -5 * torch.ones(2,))),
            torch.ones(2, 2)
        ), 1
    )
    gmm = torch.distributions.MixtureSameFamily(mix, comp)

    samples = gmm.sample([100])


    generator = Generator(input_dim = 2, output_dim=2)
    discriminator = Discriminator(input_dim=2)
    dataloader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=True, pin_memory=True)

    gan_training(discriminator, generator, dataloader)
