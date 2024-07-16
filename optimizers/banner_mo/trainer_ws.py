from copy import deepcopy

import numpy as np
import torch
from torch import distributions as dist
from torch.nn.utils import spectral_norm
from tqdm.auto import tqdm


class DistanceEstimator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            spectral_norm(torch.nn.Linear(input_dim, 256)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(torch.nn.Linear(256, 256)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(torch.nn.Linear(256, 256)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(torch.nn.Linear(256, 1)),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class EarlyStopping:
    def __init__(self, count_down=5) -> None:
        self.reset_count_down = count_down
        self.count_down = count_down
        self.current_best = np.inf
        self.update_best = False

    def _reset(self):
        self.count_down = self.reset_count_down

    def early_stop(self, current):
        if self.count_down > 0:
            if current < self.current_best:
                self.current_best = current
                self.update_best = True
                self._reset()
            else:
                self.update_best = False
                self.count_down -= 1
            return False
        else:
            return True


class TrainingRecorder:
    def __init__(self) -> None:
        self.loss_history = []

    @property
    def loss(self):
        return np.mean(self.loss_history)

    def update_loss(self, loss, **kwargs):
        self.loss_history.append(loss)


def train(
    model,
    train_data,
    num_epochs,
    batch_size,
    validation_data=None,
    shuffle=True,
    dtype=torch.float32,
    device=torch.device("cpu"),
    *args,
    **kwargs
):
    X_train, task_ids, y_train = train_data
    X_train = torch.tensor(X_train, dtype=dtype, device=device)
    y_train = torch.tensor(y_train, dtype=dtype, device=device)
    task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)

    train_tensors = [X_train, task_ids, y_train]
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    model.initialize(task_ids[-1], device)
    training_loop(
        model,
        train_dataloader,
        num_epochs,
        device
    )


def training_loop(
    model,
    train_dataloader,
    num_epochs,
    device,
    interval=1
):  
    de = DistanceEstimator(model.task_embedding_size).to(device)
    optimizer_de = torch.optim.AdamW(de.parameters(), lr=1e-3, betas=(0.5, 0.999))
    scheduler_de = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_de, gamma=0.999
    )

    early_stopping = EarlyStopping()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.999
    )

    num_features = 2
    mix = dist.Categorical(torch.ones(2, device=device))
    comp = dist.Independent(dist.Normal(
        torch.vstack((2 * torch.ones(num_features, device=device),
                     -2 * torch.ones(num_features, device=device))),
        0.5 * torch.ones(2, num_features, device=device)),
        1
    )
    prior_gmm = dist.MixtureSameFamily(mix, comp)

    de.train()
    model.train()
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            # keep track of history of only one epoch
            epoch_recorder = TrainingRecorder()
            prediction_loss_recorder = TrainingRecorder()
            embedding_loss_recorder = TrainingRecorder()

            prediction_coeff = 1 - 0.5 * epoch / num_epochs
            embedding_coeff = 1 - prediction_coeff
            # prediction_coeff = .99
            # embedding_coeff = 1 - prediction_coeff
            for _, batch_data in enumerate(train_dataloader):
                optimizer_de.zero_grad(set_to_none=True)

                # update discriminator
                prior_samples = prior_gmm.sample(model.task_embedding.weight[1:].shape[:1])
                loss_de = (de(model.task_embedding.weight[1:]).mean() -
                           de(prior_samples).mean())
                loss_de.backward()
                optimizer_de.step()

                # update actual model
                if epoch % interval == 0:
                    # inputs with x and task_id
                    optimizer.zero_grad(set_to_none=True)
                    inputs = batch_data[:][:2]
                    targets = batch_data[:][2]

                    _, predictions = model(*inputs)
                    prediction_loss = torch.nn.functional.mse_loss(
                        predictions,
                        targets
                    )
                    embedding_loss = (de(prior_samples).mean() - 
                                      de(model.task_embedding.weight[1:]).mean())
                    batch_loss = prediction_coeff *prediction_loss + embedding_coeff * embedding_loss

                    batch_loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    epoch_recorder.update_loss(batch_loss.detach().cpu().numpy())
                    prediction_loss_recorder.update_loss(prediction_loss.detach().cpu().numpy())
                    embedding_loss_recorder.update_loss(embedding_loss.detach().cpu().numpy())

            scheduler_de.step()
            scheduler.step()

            if (epoch + 1) % 1 == 0:
                pbar.set_description(f"Epoch {epoch + 1}")
                pbar.set_postfix({
                    'prediction_loss': f"{epoch_recorder.loss:.4f}",
                    'embedding_loss': f"{embedding_loss_recorder.loss:.4f}",
                    "loss": f"{epoch_recorder.loss:.4f}"
                })

            stop = early_stopping.early_stop(prediction_loss_recorder.loss)
            if early_stopping.update_best:
                # cache model
                best_model_state_dict = deepcopy(model.state_dict())
            if stop:
                model.load_state_dict(best_model_state_dict)
                print(f"... Current best loss {early_stopping.current_best:.4f}\n"
                      f"... Early Stopped at epoch {epoch}")
                break

    model.eval()
