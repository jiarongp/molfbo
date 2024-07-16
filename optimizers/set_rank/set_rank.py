import torch

from botorch.utils.transforms import unnormalize, normalize

import torch
from botorch.utils.sampling import draw_sobol_samples, sample_hypersphere
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from .models import SmallSetTransformer
from .loss import listMLE_weighted


class RandomHypervolume:
    def __init__(self, num_objectives, ref_point, device="cpu", dtype=torch.float32) -> None:
        self.tkwargs = {"dtype": dtype, "device": device}

        self.k = torch.tensor(num_objectives)
        self.ref_point = ref_point
        self.hv_weights = sample_hypersphere(d=self.k, n=1000, qmc=True).abs().unsqueeze(1).to(**self.tkwargs)
        self.c_k = torch.pow(torch.pi, self.k / 2) / (torch.pow(torch.tensor(2), self.k) * torch.lgamma(self.k/2 + 1).exp())

    def random_hypervolume(self, y):
        # dimension-independent constant
        scalar = ((y - self.ref_point).clamp_min(0).unsqueeze(-3) / self.hv_weights).amin(dim=-1).pow(self.k).amax(dim=-1)
        hv_scalar = self.c_k * scalar.mean()
        return hv_scalar.view(-1, 1)


    def hypervolume(self, y):
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=y)
        volume = bd.compute_hypervolume()
        return volume.view(-1, 1)


def inf_train_gen(dataloader):
    while True:
        for x, y in iter(dataloader): yield x, y


def gen_data(dataset, scalarizer, batch_size, **tkwargs):
    # number of x in a set
    max_length = dataset.tensors[0].shape[0]
    length = torch.randint(low=5, high=max_length + 1, size=(1,)).item()
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=length,
        shuffle=True,
        drop_last=True,
    )

    train_gen = inf_train_gen(trainloader)
    x_samples = torch.empty(0, length, dataset.tensors[0].shape[-1], **tkwargs)
    y_samples = torch.empty(0, 1, **tkwargs)
    for _ in range(batch_size):
        x, y = next(train_gen)
        x_samples = torch.concat([x_samples, x.unsqueeze(0)], dim=0)
        y_samples = torch.concat([y_samples, scalarizer.random_hypervolume(y)], dim=0)

    return x_samples, y_samples


def train_model(model, X_obs, y_obs, scalarizer, num_iter=1000):
    dataset = torch.utils.data.TensorDataset(*[X_obs, y_obs])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = listMLE_weighted
    losses = []
    batch_size = 8
    
    model.train()
    for _ in range(num_iter):
        x_samples, y_samples = gen_data(dataset=dataset, scalarizer=scalarizer, batch_size=batch_size)
        optimizer.zero_grad()

        preds = model(x_samples)
        loss = criterion(preds.view(1, -1), y_samples.view(1, -1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
#         scheduler.step()
    model.eval()
    return model, losses



class SetRank:

    def __init__(self, problem, device="cpu", dtype=torch.float64):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.n_candidates = min(5000, max(2000, 200 * problem.bounds.shape[-1]))
        self.model = SmallSetTransformer(dim_in=self.problem.dim)
        self.model.to(**self.tkwargs)

        self.scalarizer = RandomHypervolume(self.problem.num_objectives, self.problem.ref_point)

    def fit_model(self, X_obs, y_obs, num_iter):
        self.model, _ = train_model(
            self.model,
            X_obs,
            y_obs,
            scalarizer=self.scalarizer,
            num_iter=num_iter
        )

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds).to(**self.tkwargs)

        # create candidate sets
        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=1, q=self.n_candidates).squeeze(0)
        x_cand_sets = torch.cat((
            torch.tile(X_obs_norm.unsqueeze(0),(self.n_candidates, 1, 1)),
            x_cands.to(**self.tkwargs).unsqueeze(1)
        ), dim=1)

        self.fit_model(X_obs_norm, y_obs, num_iter=500)

        with torch.no_grad():
            scores = self.model(x_cand_sets)

        candidates = x_cands[scores.argmax()].unsqueeze(0)
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)

        return new_x
