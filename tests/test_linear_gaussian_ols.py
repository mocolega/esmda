import numpy as np
from esmda4d import esmda_update


def forward_model(m, G):
    """
    Linear forward model.

    Parameters
    ----------
    m : ndarray, shape (n_parameters, n_ensemble)
        Ensemble of model parameters.

    G : ndarray, shape (n_data, n_parameters)
        Linear forward operator.

    Returns
    -------
    d : ndarray, shape (n_data, n_ensemble)
        Predicted data.
    """
    return G @ m


rng = np.random.default_rng(1234)

# Number of ensemble members
Ne = 10_000

# Prior mean
m_prior_mean = np.array([0.0, 0.0])

# Prior covariance
Cm = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])

# Linear forward operator
G = np.array([
    [1.0, 0.5],
    [0.2, 1.0],
])

# Observed data
d_obs = np.array([1.0, -0.5])

# Observation-error standard deviation
sigma = np.array([0.2, 0.2])

# Observation-error covariance
Ce = np.diag(sigma**2)

# Prior ensemble
M = rng.multivariate_normal(
    mean=m_prior_mean,
    cov=Cm,
    size=Ne,
).T


D = forward_model(M, G)

print(M.shape)
print(D.shape)

Cm_inv = np.linalg.inv(Cm)
Ce_inv = np.linalg.inv(Ce)

C_post_exact = np.linalg.inv(
    Cm_inv + G.T @ Ce_inv @ G
)

m_post_exact = C_post_exact @ (
    Cm_inv @ m_prior_mean
    + G.T @ Ce_inv @ d_obs
)

print("Exact posterior mean:")
print(m_post_exact)

print("\nExact posterior covariance:")
print(C_post_exact)


alpha = [4.0, 4.0, 4.0, 4.0]

M_esmda = M.copy()

for a in alpha:

    D_esmda = forward_model(
        M_esmda,
        G,
    )

    M_esmda = esmda_update(
        M=M_esmda,
        D=D_esmda,
        d_obs=d_obs,
        Ce=Ce,
        alpha=a,
        rng=rng,
    )

    m_post_esmda = M_esmda.mean(axis=1)

C_post_esmda = np.cov(
    M_esmda,
    bias=False,
)

print("\nES-MDA posterior mean:")
print(m_post_esmda)

print("\nExact posterior mean:")
print(m_post_exact)

print("\nES-MDA posterior covariance:")
print(C_post_esmda)

print("\nExact posterior covariance:")
print(C_post_exact)