import numpy as np

import numpy as np


def constant_alphas(n_assimilations):
    """
    Generate constant ES-MDA inflation factors.

    For constant inflation factors, alpha = n_assimilations,
    which guarantees:

        sum(1 / alpha_k) = 1

    Parameters
    ----------
    n_assimilations : int
        Number of ES-MDA assimilation steps.

    Returns
    -------
    alphas : ndarray
        Inflation factors.
    """

    if not isinstance(n_assimilations, int):
        raise TypeError("n_assimilations must be an integer.")

    if n_assimilations <= 0:
        raise ValueError("n_assimilations must be positive.")

    return np.full(
        n_assimilations,
        float(n_assimilations),
    )


def validate_alphas(alphas, tol=1e-10):
    """
    Validate ES-MDA inflation factors.

    The inflation factors must satisfy:

        sum(1 / alpha_k) = 1
    """

    alphas = np.asarray(alphas, dtype=float)

    if alphas.ndim != 1:
        raise ValueError("alphas must be a one-dimensional sequence.")

    if len(alphas) == 0:
        raise ValueError("alphas cannot be empty.")

    if np.any(alphas <= 0):
        raise ValueError("all inflation factors must be positive.")

    inflation_sum = np.sum(1.0 / alphas)

    if not np.isclose(
        inflation_sum,
        1.0,
        atol=tol,
        rtol=0.0,
    ):
        raise ValueError(
            "Invalid ES-MDA inflation factors: "
            f"sum(1/alpha) = {inflation_sum:.12f}, "
            "but it must equal 1."
        )

    return alphas

def validate_alphas(alphas, tol=1e-10):
    """
    Validate ES-MDA inflation factors.

    The inflation factors must satisfy:

        sum(1 / alpha_k) = 1

    Parameters
    ----------
    alphas : array-like
        ES-MDA inflation factors.

    tol : float
        Absolute tolerance for the validation.

    Returns
    -------
    alphas : ndarray
        Validated inflation factors.

    Raises
    ------
    ValueError
        If any alpha is non-positive or if the ES-MDA
        inflation condition is not satisfied.
    """

    alphas = np.asarray(alphas, dtype=float)

    if alphas.ndim != 1:
        raise ValueError("alphas must be a one-dimensional sequence.")

    if len(alphas) == 0:
        raise ValueError("alphas cannot be empty.")

    if np.any(alphas <= 0):
        raise ValueError("all inflation factors must be positive.")

    inflation_sum = np.sum(1.0 / alphas)

    if not np.isclose(inflation_sum, 1.0, atol=tol, rtol=0.0):
        raise ValueError(
            "Invalid ES-MDA inflation factors: "
            f"sum(1/alpha) = {inflation_sum:.12f}, "
            "but it must equal 1."
        )

    return alphas