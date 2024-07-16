import torch


def gan_training(
    discriminator,
    generator,
    dataloader,
    num_epochs=50,
    num_critic=5,
    device='cpu'
):
    discriminator.train()
    generator.train()

    # Optimizers
    optimizer_g = torch.optim.AdamW(
        generator.parameters(), lr=1e-4, weight_decay=1e-4
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(), lr=4e-4, weight_decay=1e-4
    )

    for epoch in range(num_epochs):
        for i, data_real in enumerate(dataloader, 0):

            # Train with all real data
            # update discriminator network: maximize E(log(D(x))) + E(log(1 - D(G(z))))
            optimizer_d.zero_grad()

            # update generator: maximize log(D(G(z)))
            data_real = data_real.to(device)
            z = torch.randn(dataloader.batch_size, generator.input_dim)
            with torch.no_grad():
                data_fake = generator(z)
            label_real = torch.full((dataloader.batch_size,), 1.0, dtype=torch.float, device=device)
            label_fake = torch.full((dataloader.batch_size,), 0.0, dtype=torch.float, device=device)

            outputs_real = discriminator(data_real)
            outputs_fake = discriminator(data_fake)
            d_loss = - torch.mean(torch.log(outputs_real)) + torch.mean(torch.log(outputs_fake))
            d_loss.backward()

            optimizer_d.step()

            if i % num_critic == 0:
                # Train generator
                optimizer_g.zero_grad()

                data_fake = generator(z)
                g_loss = - torch.mean(discriminator(data_fake))
                g_loss.backward()
                optimizer_g.step()
