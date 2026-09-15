import numpy as np

from esmda4d import (
    esmda_assimilate,
    constant_alphas,
)


G = np.array([
    [1.0, 0.5],
    [0.2, 1.0],
])


def forward_model(M):
    return G @ M


def test_tsvd_matches_direct():

    # Problem definition
    m_prior_mean = np.array([0.0, 0.0])
    Cm = np.eye(2)

    m_true = np.array([1.5, -0.8])

    sigma = np.array([0.2, 0.2])
    Ce = np.diag(sigma**2)

    alphas = constant_alphas(4)

    # Generate synthetic observation
    rng_setup = np.random.default_rng(1234)

    d_true = G @ m_true

    obs_error = rng_setup.multivariate_normal(
        mean=np.zeros(len(d_true)),
        cov=Ce,
    )

    d_obs = d_true + obs_error

    # Generate prior ensemble
    Ne = 10_000

    M = rng_setup.multivariate_normal(
        mean=m_prior_mean,
        cov=Cm,
        size=Ne,
    ).T

    # Use identical random perturbations
    # for direct and TSVD
    rng_direct = np.random.default_rng(5678)
    rng_tsvd = np.random.default_rng(5678)

    M_direct, _ = esmda_assimilate(
        M=M,
        forward_model=forward_model,
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng_direct,
        inversion="direct",
    )

    M_tsvd, _ = esmda_assimilate(
        M=M,
        forward_model=forward_model,
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng_tsvd,
        inversion="tsvd",
        energy=0.99,
    )

    np.testing.assert_allclose(
        M_tsvd,
        M_direct,
        atol=1e-12,
        rtol=1e-12,
    )