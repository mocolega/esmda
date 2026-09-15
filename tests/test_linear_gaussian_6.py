import numpy as np
from esmda4d import esmda_assimilate, constant_alphas

def forward_model(M):
    return G @ M

def model_error(m_estimate, m_true):
    return np.linalg.norm(m_estimate - m_true)

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

# True model
m_true = np.array([1.5, -0.8])

# Noise-free true data
d_true = G @ m_true

# Observation uncertainty
sigma = np.array([0.2, 0.2])
Ce = np.diag(sigma**2)

# Generate one noisy observation
obs_error = rng.multivariate_normal(
    mean=np.zeros(len(d_true)),
    cov=Ce,
)

d_obs = d_true + obs_error

print("True model:")
print(m_true)

print("\nTrue data:")
print(d_true)

print("\nObserved data:")
print(d_obs)

print("\nObservation error:")
print(obs_error)


# Prior ensemble
M = rng.multivariate_normal(
    mean=m_prior_mean,
    cov=Cm,
    size=Ne,
).T


D = forward_model(M)


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


alphas = constant_alphas(4)

history = []

# -------------------------
# PRIOR
# -------------------------
D_prior = G @ M_esmda

data_mean_prior = D_prior.mean(axis=1)

prior_mean = M_esmda.mean(axis=1)

prior_model_error = model_error(
    prior_mean,
    m_true,
)

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

print("Distance to true model:")
print(prior_model_error)


rng_direct = np.random.default_rng(5678)
rng_tsvd = np.random.default_rng(5678)

M_direct, history_direct = esmda_assimilate(
    M=M,
    forward_model=forward_model,
    d_obs=d_obs,
    Ce=Ce,
    alphas=alphas,
    rng=rng_direct,
    inversion="direct",
)

M_tsvd, history_tsvd = esmda_assimilate(
    M=M,
    forward_model=forward_model,
    d_obs=d_obs,
    Ce=Ce,
    alphas=alphas,
    rng=rng_tsvd,
    inversion="tsvd",
    energy=0.99,
)

mean_direct = M_direct.mean(axis=1)
mean_tsvd = M_tsvd.mean(axis=1)

cov_direct = np.cov(M_direct)
cov_tsvd = np.cov(M_tsvd)

print("DIRECT MEAN:")
print(mean_direct)

print("\nTSVD MEAN:")
print(mean_tsvd)

print("\nDIFFERENCE:")
print(mean_tsvd - mean_direct)

print("\nDIRECT COVARIANCE:")
print(cov_direct)

print("\nTSVD COVARIANCE:")
print(cov_tsvd)

print("\nCOVARIANCE DIFFERENCE:")
print(cov_tsvd - cov_direct)

print("History (Direct):")
print(history_direct)

print("\nHistory (TSVD):")
print(history_tsvd)