import torch
import numpy as np

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

# The listMLE_weighted is an extended version of listMLE with weighting added to it.
def listMLE_weighted(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, **tkwargs):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    ####### Weighting extension
    # Weighted ranking because it is more important to get the the first ranks right than the rest.
    weight = np.log(np.arange(observation_loss.shape[-1]) + 2)  # Adding 2 to prevent using log(0) & log(1) as weights.
    weight = np.array(weight, dtype=np.float32)
    weight = torch.from_numpy(weight)[None, :].to(**tkwargs)
    observation_loss = observation_loss / weight
    #######

    return torch.mean(torch.sum(observation_loss, dim=1))
