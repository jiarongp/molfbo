import numpy as np
import torch
from torch import distributions as dist
from tqdm.auto import tqdm


def inverse_multiquadratics(x, y, d, sigma):
    C = 2 * d * sigma**2
    return C / (C + torch.norm(x - y))



class EarlyStopping:
    def __init__(self, count_down=20) -> None:
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
    )


def training_loop(
    model,
    train_dataloader,
    num_epochs,
):  
    early_stopping = EarlyStopping()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    mix = dist.Categorical(torch.ones(2,))
    comp = dist.Independent(dist.Normal(
        torch.vstack((torch.tensor([5., 5.]),
                        torch.tensor([-5., -5.]))),
        torch.ones(2, 2)),
        1
    )
    prior_gmm = dist.MixtureSameFamily(mix, comp)

    model.train()
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            # keep track of history of only one epoch
            epoch_recorder = TrainingRecorder()
            for _, batch_data in enumerate(train_dataloader):

                # update actual model
                optimizer.zero_grad(set_to_none=True)
                inputs = batch_data[:][:2]
                targets = batch_data[:][2]

                _, predictions = model(*inputs)
                prediction_loss = torch.nn.functional.mse_loss(
                    predictions,
                    targets
                )
                embedding_loss =  0
                batch_loss = prediction_loss + embedding_loss

                batch_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    epoch_recorder.update_loss(batch_loss.detach().cpu().numpy())

            if (epoch + 1) % 5 == 0:
                pbar.set_description(f"Epoch {epoch + 1}")
                pbar.set_postfix(loss= f"{epoch_recorder.loss:.4f}")

            # stop = early_stopping.early_stop(epoch_recorder.loss)
            # if early_stopping.update_best:
            #     # cache model
            #     best_model_state_dict = deepcopy(model.state_dict())
            # if stop:
            #     model.load_state_dict(best_model_state_dict)
            #     print(f"... Current best loss {early_stopping.current_best:.4f}\n"
            #           f"... Early Stopped at epoch {epoch}")
            #     break

    model.eval()
