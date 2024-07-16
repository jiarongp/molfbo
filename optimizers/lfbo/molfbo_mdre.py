import math
from collections import OrderedDict

import torch
import pygmo as pg
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples


class JointMLP(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_units=64,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.activation = torch.nn.ELU()

        layers = []
        layers.append((f"input_layer", torch.nn.Linear(input_dim, num_units)))
        layers.append((f"activation", self.activation))
        layers.append((f"dropout", self.dropout))
        for i in range(num_layers - 1):
            layers.append((f"layer_{i}", torch.nn.Linear(num_units, num_units)))
            layers.append((f"activation_{i}", self.activation))
            layers.append((f"dropout_{i}", self.dropout))
        layers.append((f"output_layer", torch.nn.Linear(num_units, output_dim)))

        self.model = torch.nn.Sequential(OrderedDict(layers))
        
    def forward(self, x, gamma):
        input = torch.concat([x, gamma], dim=-1)
        features = self.model[0](input)
        for i in range(1, len(self.model) - 2):
            if isinstance(self.model[i], torch.nn.Linear):
                identity = features
                features = self.model[i](features)
                features += identity
            else:
                features = self.model[i](features)
        logits = self.model[-1](features)
        return logits


class LFBO_JointRand:

    def __init__(
        self,
        input_dim,
        output_dim,
        num_units=64,
        num_layers=3,
        weight_type='pi',
        interpolation='lower',
        device="cpu:0",
        dtype=torch.double,
    ):
        self.tkwargs = {"device": device, "dtype": dtype}
        self.clf = JointMLP(
            input_dim=input_dim + 1, # plus 1 for gamma
            output_dim=output_dim,
            num_units=num_units,
            num_layers=num_layers,
        )
        self.clf.to(**self.tkwargs)
        self.weight_type = weight_type
        self.interpolation = interpolation

    @staticmethod
    def split_good_bad(X, y, x_m, gamma, weight_type, interpolation):
        tau = torch.quantile(torch.unique(y), q=gamma, interpolation=interpolation)
        z = torch.less(y, tau)
        x_p = X[z.squeeze(), :]
        x_q = X[~z.squeeze(), :]

        z_p = torch.empty(x_p.shape[0], dtype=torch.long).fill_(0)
        z_q = torch.empty(x_q.shape[0], dtype=torch.long).fill_(1)
        z_m = torch.empty(x_m.shape[0], dtype=torch.long).fill_(2)

        x = torch.cat((x_p, x_q, x_m), dim=0)
        z = torch.cat((z_p, z_q, z_m), dim=0)  
        z = torch.nn.functional.one_hot(z)

        w = torch.tensor([len(x_p), len(x_q), len(x_m)]) / len(x)
        return x, z, w 

    def fit(
        self,
        X_obs,
        y_obs,
        x_m,
        batch_size=64,
        S=100
    ):
        optimizer = torch.optim.Adam(self.clf.parameters())
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        loss_fn = torch.nn.CrossEntropyLoss()

        N = len(X_obs)  # N-th iteration
        M = math.ceil(N / batch_size)  # Steps per epochs
        E = math.floor(S / M)

        self.clf.train()
        for epochs in range(E):
            gamma = torch.rand(1, **self.tkwargs)
            new_X, new_z, new_w = self.split_good_bad(
                X_obs.detach(), y_obs.detach(), x_m, gamma=gamma, weight_type=self.weight_type, interpolation=self.interpolation
            )
            new_quantile = torch.tensor([[gamma]] * len(new_X), **self.tkwargs)

            # add batch dimension
            train_tensors = [new_X.unsqueeze(1), new_z.to(**self.tkwargs), new_quantile.unsqueeze(1)]
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            for _, (inputs, targets, quantile) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                outputs = self.clf(inputs, quantile)
                batch_loss = loss_fn(
                    outputs.squeeze(1), targets
                )
                batch_loss.backward()
                optimizer.step()
            # scheduler.step()
        self.clf.eval()

    def predict(self, X, gamma):
        quantile = torch.tensor([gamma] * X.shape[0], **self.tkwargs)

        # add batch dimension
        if len(quantile.shape) == 1:
            quantile = quantile[..., None, None]

        with torch.no_grad():
            return self.clf(X.unsqueeze(1), quantile)


def get_partial_observations(y_obs, gamma, tkwargs):
    ndf, _, _, _ = pg.core.fast_non_dominated_sorting(-y_obs.numpy())
    print(f"Number of pareto shell {len(ndf)}")

    num_obs = 0
    for n, shell in enumerate(ndf, start=1):
        num_obs += len(shell)
        if num_obs / len(y_obs) > gamma:
            break
    shell_idx = n

    y_shell = torch.empty(0, y_obs.shape[-1], **tkwargs)
    for i in range(shell_idx, len(ndf), 1):
        y_shell = torch.cat((y_shell, y_obs[ndf[i].astype(int)]))

    return y_shell


class MOLFBO_MDRE:

    def __init__(self, problem, device="cpu", dtype=torch.double):
        self.problem = problem
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.n_candidates = min(5000, max(2000, 200 * problem.bounds.shape[-1]))
        self.warm_start = True
        self.clf_list = [
            LFBO_JointRand(input_dim=problem.bounds.shape[-1], output_dim=3, weight_type='pi', **self.tkwargs)
            for _ in range(problem.num_objectives)
        ]

    def fit_model(self, X_obs, y_obs, x_m, S):
        for i, clf in enumerate(self.clf_list):
            clf.fit(X_obs=X_obs, y_obs=y_obs[:, i:i+1], x_m=x_m, S=S)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        # normalize training input
        X_obs_norm = normalize(X_obs, self.problem.bounds)
        
        y_shell = get_partial_observations(y_obs, gamma=0.1, tkwargs=self.tkwargs)

        # they assumes maximization
        bd = DominatedPartitioning(ref_point=self.problem.ref_point, Y=y_shell)
        nbd = FastNondominatedPartitioning(ref_point=self.problem.ref_point, Y=y_shell)
        _, pareto = bd.hypercell_bounds
        ndom, _ = nbd.hypercell_bounds

        x_cands = draw_sobol_samples(bounds=self.standard_bounds, n=1, q=self.n_candidates).squeeze(0)
        # negate to turn into minimization
        y_obs = -y_obs
        pareto = -pareto
        ndom = -ndom

        if self.warm_start:
            self.warm_start = False
            S = 1000
        else:
            S = 100

        x_m = draw_sobol_samples(bounds=self.problem.bounds, n=int(len(X_obs_norm)/2), q=1).squeeze(1)
        self.fit_model(X_obs_norm, y_obs, x_m, S=S)

        # u: upper non-dominated point
        # l: lower dominated point
        ref_pts = torch.concat((ndom, pareto), dim=0)
        pi_per_region = torch.empty((0, self.n_candidates, 1), **self.tkwargs)
        for ref in ref_pts:
            preds = torch.empty((0, self.n_candidates, 1), **self.tkwargs)
            for i, clf in enumerate(self.clf_list):
                gamma = ((y_obs[:, i:i+1] <= ref[i]).sum() / len(y_obs)).to(**self.tkwargs)

                with torch.no_grad():
                    logits = clf.predict(x_cands, gamma=gamma)
                    class_prob = torch.nn.functional.softmax(logits, dim=-1)
                    preds = torch.concat([preds, (1 - class_prob[:, :, 1]).unsqueeze(0)], dim=0)

            agg_preds = torch.cumprod(preds, dim=0)[-1]
            pi_per_region = torch.concat((pi_per_region, agg_preds.unsqueeze(0)), dim=0)

        # the number pareto points is always one less than the non-dominated points
        pi_per_region = torch.concat((pi_per_region, torch.zeros_like(pi_per_region[0]).unsqueeze(0)), dim=0)
        pi_per_interval = pi_per_region[:len(ndom)] - pi_per_region[len(ndom):]
        # pi = torch.sum(pi_per_region, dim=0)
        pi = torch.sum(pi_per_interval, dim=0)
        candidates = x_cands[pi.argmax()].unsqueeze(0)
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        return new_x
