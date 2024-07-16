from typing import Optional

import torch
from sklearn.ensemble import RandomForestClassifier


class LFBO_RF:

    def __init__(
        self,
        weight_type: str = 'ei',
        rf_num_trees: int = 1000,
        rf_bootstrap: bool = True,
        rf_class_weight: Optional[str] = None,
        rf_n_jobs: int = 1,
        device: str = "cpu:0",
        dtype = torch.double
    ):
        self.weight_type = weight_type
        self.clf = RandomForestClassifier(
            n_estimators=rf_num_trees,
            bootstrap=rf_bootstrap,
            class_weight=rf_class_weight,
            n_jobs=rf_n_jobs,
        )
        self.tkwargs = {"dtype": dtype, "device": device}

    @staticmethod
    def split_good_bad(X, y, gamma, weight_type='ei'):
        tau = torch.quantile(torch.unique(y), q=gamma)
        z = torch.less(y, tau)

        if len(X) > 1 and weight_type == 'ei':
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

        elif len(X) == 1 or weight_type == 'pi':
            x = X
            w = torch.ones_like(z)

        return x, y, z, w

    def fit(
        self,
        X,
        y,
        gamma=0.33,
        weight_type='ei'
    ):
        x, y, z, w = self.split_good_bad(
            X,
            y,
            gamma,
            weight_type
        )
        self.clf.fit(x.numpy(), z.numpy().ravel(), sample_weight=w.numpy().ravel())

    def predict(self, X):
        preds = self.clf.predict_proba(X.numpy())[:, 1]
        return torch.tensor(preds, **self.tkwargs)
