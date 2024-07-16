import numpy as np


def split_good_bad(X, y, gamma=0.3):
    tau = np.quantile(y, q=gamma)
    z = np.less(y, tau).flatten()
    return X, y, z
