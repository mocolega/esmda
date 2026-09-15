import numpy as np


def data_mismatch(
    d_pred,
    d_obs,
    Ce,
):
    """
    Compute normalized data mismatch.

    Parameters
    ----------
    d_pred : ndarray
        Predicted data vector with shape
        (n_data,).

    d_obs : ndarray
        Observed data vector with shape
        (n_data,).

    Ce : ndarray
        Observation-error covariance matrix with shape
        (n_data, n_data).

    Returns
    -------
    mismatch : float
        Normalized data mismatch:

        1 / (2 * n_data) *
        (d_obs - d_pred)^T
        Ce^{-1}
        (d_obs - d_pred)
    """

    d_pred = np.asarray(d_pred, dtype=float)
    d_obs = np.asarray(d_obs, dtype=float)
    Ce = np.asarray(Ce, dtype=float)

    if d_pred.ndim != 1:
        raise ValueError(
            "d_pred must be a 1D array."
        )

    if d_obs.ndim != 1:
        raise ValueError(
            "d_obs must be a 1D array."
        )

    if d_pred.shape != d_obs.shape:
        raise ValueError(
            "d_pred and d_obs must have the same shape."
        )

    Nd = len(d_obs)

    if Ce.shape != (Nd, Nd):
        raise ValueError(
            "Ce must have shape (n_data, n_data). "
            f"Expected ({Nd}, {Nd}), "
            f"but got {Ce.shape}."
        )

    if not np.all(np.isfinite(d_pred)):
        raise ValueError(
            "d_pred contains NaN or infinite values."
        )

    if not np.all(np.isfinite(d_obs)):
        raise ValueError(
            "d_obs contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Ce)):
        raise ValueError(
            "Ce contains NaN or infinite values."
        )

    residual = d_obs - d_pred

    mismatch = (
        residual.T
        @ np.linalg.solve(Ce, residual)
    ) / (2 * Nd)

    return float(mismatch)