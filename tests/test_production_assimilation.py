import numpy as np

from esmda4d import (
    ProductionData,
    SimulatedProductionData,
    build_production_observations,
    build_production_ensemble,
    constant_alphas,
    esmda_assimilate,
)


def test_production_assimilation():

    # -------------------------------------------------
    # Random generators
    # -------------------------------------------------

    rng_prior = np.random.default_rng(1234)
    rng_assimilation = np.random.default_rng(5678)

    # -------------------------------------------------
    # Ensemble
    #
    # Two model parameters and 5000 realizations.
    # -------------------------------------------------

    Ne = 5000

    m_prior_mean = np.array([
        0.0,
        0.0,
    ])

    Cm = np.eye(2)

    M_prior = rng_prior.multivariate_normal(
        mean=m_prior_mean,
        cov=Cm,
        size=Ne,
    ).T

    # -------------------------------------------------
    # True reservoir model
    # -------------------------------------------------

    m_true = np.array([
        1.5,
        -0.8,
    ])

    # -------------------------------------------------
    # Synthetic production model
    #
    # We pretend these equations represent production
    # quantities returned by a reservoir simulator.
    #
    # Oil rate:
    #
    # q_o(t0) = 1000 + 100*m1 +  50*m2
    # q_o(t1) =  900 +  80*m1 +  40*m2
    #
    # Water rate:
    #
    # q_w(t0) = 500 - 30*m1 + 60*m2
    # q_w(t1) = 600 - 20*m1 + 70*m2
    # -------------------------------------------------

    def simulate_production(M):

        oil_values = np.vstack([
            1000.0
            + 100.0 * M[0, :]
            + 50.0 * M[1, :],

            900.0
            + 80.0 * M[0, :]
            + 40.0 * M[1, :],
        ])

        water_values = np.vstack([
            500.0
            - 30.0 * M[0, :]
            + 60.0 * M[1, :],

            600.0
            - 20.0 * M[0, :]
            + 70.0 * M[1, :],
        ])

        oil = SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30],
            values=oil_values,
        )

        water = SimulatedProductionData(
            entity="FIELD",
            entity_type="field",
            variable="water_rate",
            time=[0, 30],
            values=water_values,
        )

        return [
            oil,
            water,
        ]

    # -------------------------------------------------
    # Generate synthetic true production
    # -------------------------------------------------

    M_true = m_true[:, None]

    # simulate_production() requires an ensemble with
    # at least 2 realizations because
    # SimulatedProductionData validates that condition.
    #
    # Therefore create two identical true models.

    M_true_pair = np.repeat(
        M_true,
        2,
        axis=1,
    )

    true_simulation = simulate_production(
        M_true_pair
    )

    oil_true = true_simulation[0].values[:, 0]
    water_true = true_simulation[1].values[:, 0]

    # -------------------------------------------------
    # Observed production data
    #
    # For this integration test we use the exact
    # synthetic truth as the observations.
    # -------------------------------------------------

    oil_obs = ProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30],
        values=oil_true,
        std=[10.0, 10.0],
    )

    water_obs = ProductionData(
        entity="FIELD",
        entity_type="field",
        variable="water_rate",
        time=[0, 30],
        values=water_true,
        std=[10.0, 10.0],
    )

    # -------------------------------------------------
    # Build ES-MDA observations
    # -------------------------------------------------

    d_obs, Ce, metadata = (
        build_production_observations([
            oil_obs,
            water_obs,
        ])
    )

    # -------------------------------------------------
    # ES-MDA forward model
    #
    # M
    #  ↓
    # synthetic simulator
    #  ↓
    # SimulatedProductionData
    #  ↓
    # build_production_ensemble()
    #  ↓
    # D
    # -------------------------------------------------

    def forward_model(M):

        simulated_data = simulate_production(
            M
        )

        D = build_production_ensemble(
            simulated_data=simulated_data,
            metadata=metadata,
        )

        return D

    # -------------------------------------------------
    # Prior diagnostics
    # -------------------------------------------------

    prior_mean = M_prior.mean(
        axis=1
    )

    prior_error = np.linalg.norm(
        prior_mean - m_true
    )

    # -------------------------------------------------
    # Run ES-MDA
    # -------------------------------------------------

    alphas = constant_alphas(4)

    M_final, history = esmda_assimilate(
        M=M_prior,
        forward_model=forward_model,
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng_assimilation,
        inversion="tsvd",
        energy=0.99,
    )

    # -------------------------------------------------
    # Posterior diagnostics
    # -------------------------------------------------

    posterior_mean = M_final.mean(
        axis=1
    )

    posterior_error = np.linalg.norm(
        posterior_mean - m_true
    )

    # -------------------------------------------------
    # Tests
    # -------------------------------------------------

    # Production layer must produce the expected
    # ES-MDA dimensions.

    D_prior = forward_model(
        M_prior
    )

    assert D_prior.shape == (
        4,
        Ne,
    )

    assert d_obs.shape == (4,)

    assert Ce.shape == (4, 4)

    assert len(metadata) == 4

    # ES-MDA should move the ensemble mean toward
    # the true model.

    assert (
        posterior_error
        <
        prior_error
    )

    # The final model should be reasonably close
    # to the synthetic truth.

    np.testing.assert_allclose(
        posterior_mean,
        m_true,
        atol=0.15,
        rtol=0.0,
    )

    # Data mismatch should decrease.

    assert (
        history[-1]["mismatch"]
        <
        history[0]["mismatch"]
    )

    # Four assimilations + prior.

    assert len(history) == 5

    # Verify that TSVD was actually used.

    for step in history[1:]:

        assert (
            step["inversion"]["method"]
            == "tsvd"
        )