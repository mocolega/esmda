import numpy as np
from esmda4d import esmda_update


def data_mismatch(d_pred, d_obs, Ce):
    """
    Calculate normalized data mismatch using the
    ensemble-mean predicted data.
    """

    Nd = len(d_obs)

    residual = d_obs - d_pred

    mismatch = (
        residual.T
        @ np.linalg.inv(Ce)
        @ residual
    ) / (2 * Nd)

    return mismatch

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


# ES-MDA
M_esmda = M.copy()


alphas = [4.0, 4.0, 4.0, 4.0]

history = []

# -------------------------
# PRIOR
# -------------------------
D_prior = G @ M_esmda

data_mean_prior = D_prior.mean(axis=1)

mismatch_prior = data_mismatch(
    data_mean_prior,
    d_obs,
    Ce,
)

print("\nPRIOR")
print("Mean:")
print(M_esmda.mean(axis=1))

print("Covariance:")
print(np.cov(M_esmda))

print("Predicted data mean:")
print(D_prior.mean(axis=1))

print("Observed data:")
print(d_obs)

print("Data mismatch:")
print(mismatch_prior)

history.append({
    "assimilation": 0,
    "mean": M_esmda.mean(axis=1).copy(),
    "covariance": np.cov(M_esmda).copy(),
    "data_mean": D_prior.mean(axis=1).copy(),
    "mismatch": mismatch_prior,
})


# -------------------------
# ASSIMILATIONS
# -------------------------
for i, alpha in enumerate(alphas, start=1):

    # Forward model before update
    D = G @ M_esmda

    # Assimilation
    M_esmda = esmda_update(
        M=M_esmda,
        D=D,
        d_obs=d_obs,
        Ce=Ce,
        alpha=alpha,
        rng=rng,
    )

    # Forward model after update
    D_after = G @ M_esmda

    # Statistics after assimilation
    mean_i = M_esmda.mean(axis=1)
    cov_i = np.cov(M_esmda)
    data_mean_i = D_after.mean(axis=1)

    data_mean_after = D_after.mean(axis=1)

    mismatch_after = data_mismatch(
        data_mean_after,
        d_obs,
        Ce,
    )

    print(f"\nASSIMILATION {i}")

    print("Mean:")
    print(mean_i)

    print("Covariance:")
    print(cov_i)

    print("Predicted data mean:")
    print(data_mean_i)

    print("Observed data:")
    print(d_obs)

    print("Data mismatch:")
    print(mismatch_after)

    history.append({
        "assimilation": i,
        "mean": mean_i.copy(),
        "covariance": cov_i.copy(),
        "data_mean": data_mean_i.copy(),
        "mismatch": mismatch_after,
    })


# -------------------------
# FINAL RESULTS
# -------------------------
m_post_esmda = M_esmda.mean(axis=1)
C_post_esmda = np.cov(M_esmda)

print("\nFINAL ES-MDA POSTERIOR MEAN:")
print(m_post_esmda)

print("\nEXACT POSTERIOR MEAN:")
print(m_post_exact)

print("\nFINAL ES-MDA POSTERIOR COVARIANCE:")
print(C_post_esmda)

print("\nEXACT POSTERIOR COVARIANCE:")
print(C_post_exact)