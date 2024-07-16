from typing import Optional, Generator
from contextlib import contextmanager

import torch
import numpy as np


@contextmanager
def manual_seed(seed: Optional[int]=None) -> Generator[None, None, None]:
    """
    Contextmanager for manual setting the torch.random seed.
    source: https://github.com/pytorch/botorch/blob/3f89c2966b323a501b7a01672519e455da2b5370/botorch/utils/sampling.py#L34
    Args:
        seed: The seed to set the random number generator to.
    Returns:
        Generator
    Example:
        >>> with manual_seed(1234):
        >>>     X = torch.rand(3)  
    """
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def metadata_to_training(meta_data):
    """Convert data to standard (X, y) input-output format.

    Parameters
    ----------
    metadata : dict
        Dictionary of task IDs that contain the data.
        See `metafidelity.benchmarks.base.BaseBenchmark.get_meta_data`
        for a detailed description of the output.

    Returns
    -------
    X : list of ndarrays
        Training data, with one array per task in the list.
    y : list of ndarrays
        Training data, with one array per task in the list.
    """
    # Transform metadata into a list of arrays,
    X = []
    y = []
    task_ids = []
    obj_ids = []
    for (obj_id, task_id), data in meta_data.items():
        num_obs = len(data["X"])
        X.extend(data["X"])
        y.extend(data["Y"])
        task_ids.extend(np.ones(num_obs, dtype=int) * task_id)
        obj_ids.extend(np.ones(num_obs, dtype=int) * obj_id)

    return np.array(X), np.array(obj_ids), np.array(task_ids), np.array(y)
