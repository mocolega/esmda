import numpy as np

from esmda4d import (
    esmda_assimilate,
    constant_alphas,
)


def build_linear_problem():

    Nm = 5
    Nd = 20
    Ne = 1000

    rng_setup = np.random.default_rng(1234)

    G = rng_setup.normal(
        size=(Nd, Nm)
    )

    def forward_model(M):
        return G @ M

    m_prior_mean = np.zeros(Nm)
    Cm = np.eye(Nm)

    m_true = np.array([
        1.0,
        -0.5,
        0.8,
        -1.2,
        0.3,
    ])

    d_true = G @ m_true

    sigma = np.full(Nd, 0.2)
    Ce = np.diag(sigma**2)

    obs_error = rng_setup.multivariate_normal(
        mean=np.zeros(Nd),
        cov=Ce,
    )

    d_obs = d_true + obs_error

    M = rng_setup.multivariate_normal(
        mean=m_prior_mean,
        cov=Cm,
        size=Ne,
    ).T

    return (
        M,
        forward_model,
        m_true,
        d_obs,
        Ce,
    )


def test_tsvd_larger_linear_system():

    (
        M,
        forward_model,
        m_true,
        d_obs,
        Ce,
    ) = build_linear_problem()

    alphas = constant_alphas(4)

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
        energy=0.999999,
    )

    # Full TSVD should reproduce direct inversion
    np.testing.assert_allclose(
        M_tsvd,
        M_direct,
        atol=1e-10,
        rtol=1e-10,
    )

    # Check diagnostics
    for step in history_tsvd[1:]:

        info = step["inversion"]

        assert info["method"] == "tsvd"
        assert info["n_retained"] == 5
        assert info["energy"] == 0.999999

        singular_values = info["singular_values"]

        assert singular_values.ndim == 1
        assert len(singular_values) == 20

    # Direct inversion should also report its method
    for step in history_direct[1:]:

        assert (
            step["inversion"]["method"]
            == "direct"
        )

    # Posterior should improve relative to prior
    prior_mean = M.mean(axis=1)
    posterior_mean = M_tsvd.mean(axis=1)

    prior_error = np.linalg.norm(
        prior_mean - m_true
    )

    posterior_error = np.linalg.norm(
        posterior_mean - m_true
    )

    assert posterior_error < prior_error


def test_tsvd_with_truncation():

    (
        M,
        forward_model,
        m_true,
        d_obs,
        Ce,
    ) = build_linear_problem()

    alphas = constant_alphas(4)

    rng_tsvd = np.random.default_rng(5678)

    M_tsvd, history = esmda_assimilate(
        M=M,
        forward_model=forward_model,
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng_tsvd,
        inversion="tsvd",
        energy=0.80,
    )

    retained = [
        step["inversion"]["n_retained"]
        for step in history[1:]
    ]

    assert retained == [4, 3, 4, 4]

    # Every assimilation should identify itself
    # as a TSVD update
    for step in history[1:]:

        info = step["inversion"]

        assert info["method"] == "tsvd"
        assert info["energy"] == 0.80

    # Verify actual truncation happened
    assert all(
        n_retained < 5
        for n_retained in retained
    )

    # Posterior should still improve
    prior_mean = M.mean(axis=1)
    posterior_mean = M_tsvd.mean(axis=1)

    prior_error = np.linalg.norm(
        prior_mean - m_true
    )

    posterior_error = np.linalg.norm(
        posterior_mean - m_true
    )

    assert posterior_error < prior_error