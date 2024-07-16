from typing import Union

import torch


def _marginalized_blr_assert_input_shapes(
    phi: torch.Tensor = None, hist_phi: torch.Tensor = None, hist_y: torch.Tensor = None
):
    """Assert appropriate shapes for the inputs.

    :param phi: Features of Test points
    :param hist_phi: Historical features
    :param hist_y: Historical y values
    """

    assert (
        phi is None or phi.ndim == 2
    ), f"Provided features matrix phi has to be 2D but is {phi.ndim}D"

    assert (
        hist_phi is None or hist_phi.ndim == 2
    ), f"Provided historical features matrix phi has to be 2D but is {hist_phi.ndim}D"

    assert (
        hist_y is None or hist_y.ndim == 2
    ), f"Provided historical y values matrix phi has to be 2D but is {hist_y.ndim}D"


def _jittered_cholesky(matrix, jitter=1e-10):
    """Compute cholesky of input with adaptive jitter for increased stability"""
    while True:
        try:
            L = torch.linalg.cholesky(
                matrix
                + torch.eye(matrix.size(0), device=matrix.device)
                * matrix.diag().mean()
                * jitter,
            )
            break
        except RuntimeError:
            # increase jitter, if Cholesky fails due to numerical instability
            jitter *= 10
            if jitter > 1e-1:
                raise
    return L


def marginalized_blr_predict(
    phi: torch.Tensor,
    hist_phi: torch.Tensor,
    hist_y: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float]
):
    """Predict mean and variance of an observation based on the features
       and historic data passed.

    :param phi: Features of Test points
    :param hist_phi: Historical features
    :param hist_y: Historical y values
    :param alpha: precision of prior gaussian distribution over weights
    :param beta: precision of gaussian distribution over output given NN and BLR weights

    NOTE: `phi`, `hist_phi` and `hist_y` need to be 2D tensors
    """
    _marginalized_blr_assert_input_shapes(phi, hist_phi, hist_y)

    if hist_phi.size(0) == 0:
        # if no history data is available, we must use the prior
        var = (1 / alpha) * torch.norm(phi, dim=1, p=2) ** 2 + 1 / beta
        var = var.unsqueeze(1)
        m = torch.zeros_like(var)
        return m, var

    # TODO: Implement the other case in paper
    eye = torch.eye(hist_phi.size(1)).type_as(hist_phi)

    K = (beta / alpha) * torch.mm(hist_phi.t(), hist_phi) + eye
    L = torch.linalg.cholesky(K + eye * K.diag().mean() * 1e-4)

    e = torch.linalg.solve_triangular(L, torch.mm(hist_phi.t(), hist_y), upper=False)

    temp = torch.linalg.solve_triangular(L, phi.t(), upper=False)
    mean = (beta / alpha) * torch.mm(e.t(), temp)

    # TODO: Why adding the 1 / beta at the end? There is no 1/beta in the paper.
    var = (1 / alpha) * torch.norm(temp, dim=0, p=2) ** 2 + 1 / beta
    m = mean.mean(0)

    m = m.unsqueeze(1)
    var = var.unsqueeze(1)

    return m, var


def marginalized_blr_nll_naive(
    phi: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float]
):
    """Naive expression for the NLL, which does not exploit the positive-definiteness
       of the covariance matrix.

    For small enough matrices, this expression should yield a good enough result to
    compare to the more stable versions.

    :param phi: Input values.
    :param targets: Target values from the dataset
    :param alpha: precision of prior gaussian distribution over BLR weights
    :param beta: precision of Gaussian distribution over output, i.e. inverse of
                 observational noise variance
    :return: Negative log likelihood (up to a constant)
    """
    Sigma = torch.mm(phi, phi.t()) / alpha + torch.eye(phi.size(0)) / beta
    # returns the solution to the system of linear equations represented by
    # AX = B
    tmp = torch.linalg.solve(Sigma, targets)
    # torch.logdet calculates log determinant of a square matrix
    # TODO: Why is this the nll?
    nll = (torch.logdet(Sigma) + torch.mm(targets.t(), tmp)) / 2
    return nll


def marginalized_blr_nll(
    phi: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float]
):
    """Determine and call appropriate NLL function based on the number of points vs
       features.

    :param phi: Input values.
    :param targets: Target values from the dataset
    :param alpha: precision of prior gaussian distribution over BLR weights
    :param beta: precision of Gaussian distribution over output, i.e. inverse of
                 observational noise variance
    :return: Negative log likelihood (up to a constant)
    """
    N_points, N_features = phi.size()
    if N_points < N_features:
        return _marginalized_blr_nll_fewer_points_than_features(
            phi, targets, alpha, beta
        )
    else:
        return _marginalized_blr_nll_more_points_than_features(
            phi, targets, alpha, beta
        )


def _marginalized_blr_nll_more_points_than_features(
    phi: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float]
):
    """Returns the negative log likelihood of data phi corresponding targets.

     This function is more robust and efficient if more points than features are given.
     In this case, the naive implementation often fails, as the system is overdetermined
     for small noise levels.

    :param phi: Input values.
    :param targets: Target values from the dataset
    :param alpha: precision of prior gaussian distribution over BLR weights
    :param beta: precision of Gaussian distribution over output, i.e. inverse of
                 observational noise variance
    :return: Negative log likelihood (up to a constant)
    """
    _marginalized_blr_assert_input_shapes(None, phi, targets)

    # have a look at the derivation of this in the supplementary material of the ABLR
    # paper
    r_t = beta / alpha
    eye = torch.eye(phi.size(1), device=phi.device)
    K_t = r_t * torch.mm(phi.t(), phi) + eye
    L_t = _jittered_cholesky(K_t)
    e = torch.linalg.solve_triangular(L_t, torch.mm(phi.t(), targets), upper=False)
    diff = 0.5 * beta * (torch.norm(targets, p=2) ** 2 - r_t * torch.norm(e, p=2) ** 2)
    nll = (
        - 0.5 * targets.size(0) * torch.log(beta)
        + diff
        + torch.sum(torch.log(L_t.diag()))
    )

    return nll


def _marginalized_blr_nll_fewer_points_than_features(
    phi: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float]
):
    """Returns the negative log likelihood of data phi corresponding targets.

     This function is more efficient if fewer points than features are given.

    :param phi:  Input values.
    :param targets: Target values from the dataset
    :param alpha: precision of prior gaussian distribution over BLR weights
    :param beta: precision of Gaussian distribution over output,
                 i.e. inverse of observational noise variance
    :return: Negative log likelihood (up to a constant)
    """
    _marginalized_blr_assert_input_shapes(None, phi, targets)

    r_t = beta / alpha
    eye = torch.eye(targets.size(0), device=phi.device)
    K_t = r_t * torch.mm(phi, phi.t()) + eye
    L_t = _jittered_cholesky(K_t)
    e = torch.linalg.solve_triangular(L_t, targets, upper=False)
    diff = 0.5 * beta * torch.norm(e, p=2) ** 2
    nll = (
        -0.5 * targets.size(0) * torch.log(beta)
        + diff
        + torch.sum(torch.log(L_t.diag()))
    )

    return nll
