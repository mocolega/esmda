import numpy as np
from .inflation import validate_alphas
from .inversion import tsvd_update
from .qc import data_mismatch


# def data_mismatch(d_pred, d_obs, Ce):
#     Nd = len(d_obs)
#     residual = d_obs - d_pred

#     return (
#         residual.T
#         @ np.linalg.inv(Ce)
#         @ residual
#     ) / (2 * Nd)

def _validate_update_inputs(
    M,
    D,
    d_obs,
    Ce,
    alpha,
):
    """
    Validate inputs for one ES-MDA update.

    Parameters
    ----------
    M : ndarray
        Model ensemble with shape
        (n_model_parameters, n_ensemble).

    D : ndarray
        Predicted-data ensemble with shape
        (n_data, n_ensemble).

    d_obs : ndarray
        Observed-data vector with shape
        (n_data,).

    Ce : ndarray
        Observation-error covariance matrix with shape
        (n_data, n_data).

    alpha : float
        ES-MDA inflation factor.

    Raises
    ------
    ValueError
        If dimensions are inconsistent or values are invalid.
    """

    # -------------------------------------------------
    # Number of dimensions
    # -------------------------------------------------

    if M.ndim != 2:
        raise ValueError(
            "M must be a 2D array with shape "
            "(n_model_parameters, n_ensemble)."
        )

    if D.ndim != 2:
        raise ValueError(
            "D must be a 2D array with shape "
            "(n_data, n_ensemble)."
        )

    if d_obs.ndim != 1:
        raise ValueError(
            "d_obs must be a 1D array with shape "
            "(n_data,)."
        )

    if Ce.ndim != 2:
        raise ValueError(
            "Ce must be a 2D covariance matrix."
        )

    # -------------------------------------------------
    # Extract dimensions
    # -------------------------------------------------

    Nm, Ne = M.shape
    Nd, Ne_D = D.shape

    # -------------------------------------------------
    # Ensemble size
    # -------------------------------------------------

    if Ne < 2:
        raise ValueError(
            "The ensemble must contain at least "
            "2 realizations."
        )

    # -------------------------------------------------
    # M and D must use the same ensemble size
    # -------------------------------------------------

    if Ne_D != Ne:
        raise ValueError(
            "M and D must contain the same number "
            "of ensemble realizations. "
            f"Got M.shape={M.shape} and D.shape={D.shape}."
        )

    # -------------------------------------------------
    # Observation dimensions
    # -------------------------------------------------

    if len(d_obs) != Nd:
        raise ValueError(
            "d_obs length must match the number "
            "of rows in D. "
            f"Got len(d_obs)={len(d_obs)} "
            f"and D.shape={D.shape}."
        )

    if Ce.shape != (Nd, Nd):
        raise ValueError(
            "Ce must have shape (n_data, n_data). "
            f"Expected ({Nd}, {Nd}), "
            f"but got {Ce.shape}."
        )
    # -------------------------------------------------
    # Observation-error covariance validity
    # -------------------------------------------------

    if not np.allclose(Ce, Ce.T):
        raise ValueError(
            "Ce must be symmetric."
        )

    if np.any(np.diag(Ce) <= 0):
        raise ValueError(
            "Ce diagonal entries must be positive."
        )
    # -------------------------------------------------
    # Inflation factor
    # -------------------------------------------------

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    # -------------------------------------------------
    # Numerical validity
    # -------------------------------------------------

    if not np.all(np.isfinite(M)):
        raise ValueError(
            "M contains NaN or infinite values."
        )

    if not np.all(np.isfinite(D)):
        raise ValueError(
            "D contains NaN or infinite values."
        )

    if not np.all(np.isfinite(d_obs)):
        raise ValueError(
            "d_obs contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Ce)):
        raise ValueError(
            "Ce contains NaN or infinite values."
        )

def esmda_update(
    M,
    D,
    d_obs,
    Ce,
    alpha,
    rng,
    inversion="direct",
    energy=0.99,
):
    """
    Perform one ES-MDA assimilation step.

    Parameters
    ----------
    M : ndarray, shape (n_parameters, n_ensemble)
        Current model ensemble.

    D : ndarray, shape (n_data, n_ensemble)
        Predicted-data ensemble.

    d_obs : ndarray, shape (n_data,)
        Observed data.

    Ce : ndarray, shape (n_data, n_data)
        Observation-error covariance matrix.

    alpha : float
        ES-MDA inflation factor.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    M_updated : ndarray
        Updated model ensemble.
    """

    _validate_update_inputs(
        M=M,
        D=D,
        d_obs=d_obs,
        Ce=Ce,
        alpha=alpha,
    )

    Ne = M.shape[1]

    # Ensemble means
    M_mean = M.mean(axis=1, keepdims=True)
    D_mean = D.mean(axis=1, keepdims=True)

    # Scaled ensemble anomalies
    scale = np.sqrt(Ne - 1)

    dM = (M - M_mean) / scale
    dD = (D - D_mean) / scale

    # Perturbed observations
    errors = rng.multivariate_normal(
        mean=np.zeros(len(d_obs)),
        cov=Ce,
        size=Ne,
    ).T

    D_perturbed = (
        d_obs[:, None]
        + np.sqrt(alpha) * errors
    )

    residual = D_perturbed - D

    if inversion == "direct":

        Cmd = dM @ dD.T
        Cdd = dD @ dD.T

        K = Cmd @ np.linalg.inv(
            Cdd + alpha * Ce
        )

        M_updated = M + K @ residual

        info = {
            "method": "direct",
        }

    elif inversion == "tsvd":

        increment, info = tsvd_update(
            dM=dM,
            dD=dD,
            residual=residual,
            Ce=Ce,
            alpha=alpha,
            energy=energy,
        )

        M_updated = M + increment

        info["method"] = "tsvd"
        

    else:
        raise ValueError(
            f"Unknown inversion method: {inversion}"
        )

    return M_updated, info


def esmda_assimilate(
    M,
    forward_model,
    d_obs,
    Ce,
    alphas,
    rng,
    inversion="direct",
    energy=0.99,
):
    """
    Perform a complete ES-MDA assimilation sequence.

    Parameters
    ----------
    M : ndarray, shape (n_parameters, n_ensemble)
        Initial ensemble.

    forward_model : callable
        Function that receives M and returns predicted data D.

    d_obs : ndarray, shape (n_data,)
        Observed data.

    Ce : ndarray, shape (n_data, n_data)
        Observation-error covariance.

    alphas : sequence of float
        Inflation factors.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    M : ndarray
        Final updated ensemble.

    history : list of dict
        Statistics after each assimilation.
    """

    alphas = validate_alphas(alphas)

    M = M.copy()

    history = []

    # Prior
    D = forward_model(M)

    data_mean = D.mean(axis=1)

    history.append({
        "assimilation": 0,
        "mean": M.mean(axis=1).copy(),
        "covariance": np.cov(M).copy(),
        "data_mean": data_mean.copy(),
        "mismatch": data_mismatch(
            data_mean,
            d_obs,
            Ce,
        ),
        "inversion": "prior",
    })


    # Assimilation steps
    for i, alpha in enumerate(alphas, start=1):

        D = forward_model(M)

        M, update_info = esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
            inversion=inversion,
            energy=energy,
        )

        D_after = forward_model(M)

        data_mean = D_after.mean(axis=1)

        history.append({
            "assimilation": i,
            "mean": M.mean(axis=1).copy(),
            "covariance": np.cov(M).copy(),
            "data_mean": data_mean.copy(),
            "mismatch": data_mismatch(
                data_mean,
                d_obs,
                Ce,
            ),
            "inversion": update_info,
        })

    return M, history