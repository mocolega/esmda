import numpy as np


def tsvd_update(
    dM,
    dD,
    residual,
    Ce,
    alpha,
    energy=0.99,
):
    """
    Compute the ES-MDA model increment using TSVD.

    Parameters
    ----------
    dM : ndarray
        Scaled model anomalies with shape
        (n_model_parameters, n_ensemble).

    dD : ndarray
        Scaled data anomalies with shape
        (n_data, n_ensemble).

    residual : ndarray
        Data residual matrix with shape
        (n_data, n_ensemble).

    Ce : ndarray
        Observation-error covariance matrix.

    alpha : float
        ES-MDA inflation factor.

    energy : float
        Fraction of the cumulative singular-value sum retained.

    Returns
    -------
    increment : ndarray
        Model update with shape
        (n_model_parameters, n_ensemble).
    """

    if not 0 < energy <= 1:
        raise ValueError(
            "energy must be in the interval (0, 1]."
        )
    # For now, TSVD assumes diagonal Ce
    if not np.allclose(Ce, np.diag(np.diag(Ce))):
        raise ValueError(
            "TSVD currently requires a diagonal Ce."
        )

    sigma_e = np.sqrt(np.diag(Ce))

    if np.any(sigma_e <= 0):
        raise ValueError(
            "Observation-error standard deviations "
            "must be positive."
        )

    # Scale data anomalies by observation error
    dD_scaled = dD / sigma_e[:, None]

    # Singular value decomposition
    U, s, Vt = np.linalg.svd(
        dD_scaled,
        full_matrices=False,
    )

    # Determine truncation level
    cumulative_energy = np.cumsum(s) / np.sum(s)

    Nr = (
        np.searchsorted(
            cumulative_energy,
            energy,
        )
        + 1
    )

    # Truncate
    Ur = U[:, :Nr]
    sr = s[:Nr]
    Vr = Vt[:Nr, :].T

    # Scale residual by observation error
    residual_scaled = (
        residual / sigma_e[:, None]
    )

    # TSVD coefficients:
    # sigma_i / (sigma_i^2 + alpha)
    factors = sr / (sr**2 + alpha)

    # Model increment
    increment = (
        dM
        @ Vr
        @ (
            factors[:, None]
            * (Ur.T @ residual_scaled)
        )
    )

    info = {
    "singular_values": s,
    "n_retained": Nr,
    "energy": energy,
    }
    
    return increment, info