import numpy as np

from esmda4d import (
    esmda_assimilate,
    constant_alphas,
)


def build_linear_gaussian_problem():
    """
    Build a small linear-Gaussian inverse problem with an
    analytical posterior solution.
    """

    # -------------------------------------------------
    # Reproducible random generators
    # -------------------------------------------------

    rng_problem = np.random.default_rng(1234)
    rng_assimilation = np.random.default_rng(5678)

    # -------------------------------------------------
    # Problem dimensions
    # -------------------------------------------------

    Ne = 10_000

    # -------------------------------------------------
    # Prior model
    #
    # m ~ N(m0, Cm)
    # -------------------------------------------------

    m0 = np.array([
        0.0,
        0.0,
    ])

    Cm = np.eye(2)

    # -------------------------------------------------
    # Linear forward model
    #
    # d = G m
    # -------------------------------------------------

    G = np.array([
        [1.0, 0.5],
        [0.2, 1.0],
    ])

    def forward_model(M):
        return G @ M

    # -------------------------------------------------
    # True model
    # -------------------------------------------------

    m_true = np.array([
        1.5,
        -0.8,
    ])

    d_true = G @ m_true

    # -------------------------------------------------
    # Observation-error covariance
    # -------------------------------------------------

    sigma = np.array([
        0.2,
        0.2,
    ])

    Ce = np.diag(
        sigma**2
    )

    # -------------------------------------------------
    # Synthetic observations
    # -------------------------------------------------

    observation_error = rng_problem.multivariate_normal(
        mean=np.zeros(2),
        cov=Ce,
    )

    d_obs = (
        d_true
        + observation_error
    )

    # -------------------------------------------------
    # Prior ensemble
    # -------------------------------------------------

    M_prior = rng_problem.multivariate_normal(
        mean=m0,
        cov=Cm,
        size=Ne,
    ).T

    # -------------------------------------------------
    # Analytical posterior
    # -------------------------------------------------

    Cm_inv = np.linalg.inv(Cm)
    Ce_inv = np.linalg.inv(Ce)

    C_post = np.linalg.inv(
        Cm_inv
        + G.T @ Ce_inv @ G
    )

    m_post = (
        C_post
        @ (
            Cm_inv @ m0
            + G.T @ Ce_inv @ d_obs
        )
    )

    return {
        "M_prior": M_prior,
        "forward_model": forward_model,
        "d_obs": d_obs,
        "Ce": Ce,
        "m_true": m_true,
        "m_post": m_post,
        "C_post": C_post,
        "rng_assimilation": rng_assimilation,
    }


def test_direct_linear_gaussian_matches_analytical_posterior():

    problem = build_linear_gaussian_problem()

    alphas = constant_alphas(4)

    M_final, history = esmda_assimilate(
        M=problem["M_prior"],
        forward_model=problem["forward_model"],
        d_obs=problem["d_obs"],
        Ce=problem["Ce"],
        alphas=alphas,
        rng=problem["rng_assimilation"],
        inversion="direct",
    )

    mean_final = M_final.mean(
        axis=1
    )

    covariance_final = np.cov(
        M_final
    )

    # Ensemble approximation should be close
    # to the analytical Gaussian posterior.

    np.testing.assert_allclose(
        mean_final,
        problem["m_post"],
        atol=0.02,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        covariance_final,
        problem["C_post"],
        atol=0.01,
        rtol=0.0,
    )

    # There should be:
    #
    # prior + 4 assimilation states

    assert len(history) == 5

    # Posterior mismatch should be lower than
    # prior mismatch.

    assert (
        history[-1]["mismatch"]
        <
        history[0]["mismatch"]
    )


def test_tsvd_linear_gaussian_matches_analytical_posterior():

    problem = build_linear_gaussian_problem()

    alphas = constant_alphas(4)

    M_final, history = esmda_assimilate(
        M=problem["M_prior"],
        forward_model=problem["forward_model"],
        d_obs=problem["d_obs"],
        Ce=problem["Ce"],
        alphas=alphas,
        rng=problem["rng_assimilation"],
        inversion="tsvd",
        energy=0.99,
    )

    mean_final = M_final.mean(
        axis=1
    )

    covariance_final = np.cov(
        M_final
    )

    np.testing.assert_allclose(
        mean_final,
        problem["m_post"],
        atol=0.02,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        covariance_final,
        problem["C_post"],
        atol=0.01,
        rtol=0.0,
    )

    assert len(history) == 5

    assert (
        history[-1]["mismatch"]
        <
        history[0]["mismatch"]
    )

    # Check TSVD diagnostics.

    for step in history[1:]:

        assert (
            step["inversion"]["method"]
            == "tsvd"
        )

        assert (
            step["inversion"]["n_retained"]
            == 2
        )


def test_direct_and_tsvd_give_same_result():

    problem_direct = (
        build_linear_gaussian_problem()
    )

    problem_tsvd = (
        build_linear_gaussian_problem()
    )

    alphas = constant_alphas(4)

    M_direct, _ = esmda_assimilate(
        M=problem_direct["M_prior"],
        forward_model=problem_direct[
            "forward_model"
        ],
        d_obs=problem_direct["d_obs"],
        Ce=problem_direct["Ce"],
        alphas=alphas,
        rng=problem_direct[
            "rng_assimilation"
        ],
        inversion="direct",
    )

    M_tsvd, _ = esmda_assimilate(
        M=problem_tsvd["M_prior"],
        forward_model=problem_tsvd[
            "forward_model"
        ],
        d_obs=problem_tsvd["d_obs"],
        Ce=problem_tsvd["Ce"],
        alphas=alphas,
        rng=problem_tsvd[
            "rng_assimilation"
        ],
        inversion="tsvd",
        energy=0.99,
    )

    np.testing.assert_allclose(
        M_tsvd,
        M_direct,
        atol=1e-10,
        rtol=0.0,
    )


def test_tsvd_mismatch_decreases_monotonically():

    problem = build_linear_gaussian_problem()

    alphas = constant_alphas(4)

    _, history = esmda_assimilate(
        M=problem["M_prior"],
        forward_model=problem["forward_model"],
        d_obs=problem["d_obs"],
        Ce=problem["Ce"],
        alphas=alphas,
        rng=problem["rng_assimilation"],
        inversion="tsvd",
        energy=0.99,
    )

    mismatches = [
        step["mismatch"]
        for step in history
    ]

    assert all(
        later < earlier
        for earlier, later in zip(
            mismatches[:-1],
            mismatches[1:],
        )
    )