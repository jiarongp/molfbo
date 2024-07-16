# from https://github.com/facebookresearch/morbo/blob/f0dbd6ac3b478abbf2cb6eeb1f534e9df59a9461/morbo/state.py#L815

import torch


def compute_hv_scalarizations(ref_point, Y: torch.Tensor, hv_weights: torch.Tensor) -> torch.Tensor:
    r"""Compute HV scalarizations.

    This function approximate the hypervolume with the following formula:

    $$ \mathcal{HV}_{z}(Y) := c_k \mathbb{E}_{\lambda \sim \mathcal{S}_{+}^{k-1}} \left[ \max_{y \in Y} s_{\lambda}(y - z) \right] $$

    where $s_{\lambda}(y) = \min_i(\max(0, \frac{y_i}{\lambda_i}))^k$
    and $c_k = \frac{\pi^{k/2}}{2^k \Gamma (k / 2+1)}$.
    $ \mathcal{S}_{+}^{k-1} = \{ \lambda \in \mathbb R^k \mid \lVert \lambda \rVert= 1, \lambda \geq 0 \} $ (from a hypersphere)

    Args:
        Y: A `sample_shape x batch_shape x n x m`-dim tensor of outcomes

    Returns:
        A `sample_shape x batch_shape x n_weights`-dim tensor of hv scalarizations.

    """
    # num_objectives
    k = Y.shape[-1]
    c_k = torch.pow(torch.pi, k / 2) / (torch.pow(torch.tensor(2), k) * torch.lgamma(k/2 + 1).exp())

    return (
        ((Y - ref_point).clamp_min(0).unsqueeze(-3) / hv_weights)
        .amin(dim=-1)
        .pow(Y.shape[-1])
        .amax(dim=-1)
    )