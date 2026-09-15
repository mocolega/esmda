import numpy as np
import pytest

from esmda4d.production import (
    ProductionData,
    SimulatedProductionData,
    build_production_observations,
    build_production_ensemble,
)


def build_valid_production_data():
    """
    Return valid inputs for a ProductionData object.
    """

    entity = "PROD-01"
    entity_type = "well"
    variable = "oil_rate"

    time = np.array([
        0.0,
        30.0,
        60.0,
        90.0,
    ])

    values = np.array([
        1000.0,
        950.0,
        900.0,
        850.0,
    ])

    std = np.array([
        50.0,
        50.0,
        50.0,
        50.0,
    ])

    return (
        entity,
        entity_type,
        variable,
        time,
        values,
        std,
    )


def test_valid_production_data():

    (
        entity,
        entity_type,
        variable,
        time,
        values,
        std,
    ) = build_valid_production_data()

    data = ProductionData(
        entity=entity,
        entity_type=entity_type,
        variable=variable,
        time=time,
        values=values,
        std=std,
    )

    assert data.entity == "PROD-01"
    assert data.entity_type == "well"
    assert data.variable == "oil_rate"

    np.testing.assert_array_equal(
        data.time,
        time,
    )

    np.testing.assert_array_equal(
        data.values,
        values,
    )

    np.testing.assert_array_equal(
        data.std,
        std,
    )


def test_lists_are_converted_to_numpy_arrays():

    data = ProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[1000, 950, 900],
        std=[50, 50, 50],
    )

    assert isinstance(
        data.time,
        np.ndarray,
    )

    assert isinstance(
        data.values,
        np.ndarray,
    )

    assert isinstance(
        data.std,
        np.ndarray,
    )


def test_time_must_be_1d():

    with pytest.raises(
        ValueError,
        match="time must be a 1D array",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[[0, 30], [60, 90]],
            values=[1000, 950, 900, 850],
            std=[50, 50, 50, 50],
        )


def test_values_must_be_1d():

    with pytest.raises(
        ValueError,
        match="values must be a 1D array",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30],
            values=[
                [1000, 950],
            ],
            std=[50, 50],
        )


def test_std_must_be_1d():

    with pytest.raises(
        ValueError,
        match="std must be a 1D array",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30],
            values=[1000, 950],
            std=[
                [50, 50],
            ],
        )


def test_time_values_and_std_must_have_same_length():

    with pytest.raises(
        ValueError,
        match="time, values, and std must have the same length",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[1000, 950],
            std=[50, 50],
        )


def test_production_data_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="ProductionData cannot be empty",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[],
            values=[],
            std=[],
        )


@pytest.mark.parametrize(
    "invalid_std",
    [
        [50.0, 0.0, 50.0],
        [50.0, -1.0, 50.0],
    ],
)
def test_std_must_be_positive(
    invalid_std,
):

    with pytest.raises(
        ValueError,
        match="standard deviations must be positive",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[1000, 950, 900],
            std=invalid_std,
        )


def test_values_cannot_contain_nan():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[
                1000,
                np.nan,
                900,
            ],
            std=[50, 50, 50],
        )


def test_values_cannot_contain_infinity():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[
                1000,
                np.inf,
                900,
            ],
            std=[50, 50, 50],
        )


def test_std_cannot_contain_nan():

    with pytest.raises(
        ValueError,
        match="std contains NaN or infinite values",
    ):
        ProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[1000, 950, 900],
            std=[
                50,
                np.nan,
                50,
            ],
        )


def test_std_cannot_contain_infinity():

    with pytest.raises(
        ValueError,
        match="std contains NaN or infinite values",
    ):
        ProductionData(
            entity="FIELD",
            entity_type="field",
            variable="water_rate",
            time=[0, 30, 60],
            values=[500, 600, 700],
            std=[
                25,
                np.inf,
                25,
            ],
        )

def test_build_production_observations():

    oil = ProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[1000, 950, 900],
        std=[50, 50, 50],
    )

    water = ProductionData(
        entity="FIELD",
        entity_type="field",
        variable="water_rate",
        time=[0, 30],
        values=[500, 600],
        std=[25, 25],
    )

    d_obs, Ce, metadata = (
        build_production_observations([
            oil,
            water,
        ])
    )

    # ---------------------------------------------
    # Observation vector
    # ---------------------------------------------

    expected_d_obs = np.array([
        1000.0,
        950.0,
        900.0,
        500.0,
        600.0,
    ])

    np.testing.assert_array_equal(
        d_obs,
        expected_d_obs,
    )

    # ---------------------------------------------
    # Observation-error covariance
    # ---------------------------------------------

    expected_Ce = np.diag([
        50.0**2,
        50.0**2,
        50.0**2,
        25.0**2,
        25.0**2,
    ])

    np.testing.assert_array_equal(
        Ce,
        expected_Ce,
    )

    # ---------------------------------------------
    # Metadata
    # ---------------------------------------------

    assert len(metadata) == 5

    assert metadata[0] == {
        "entity": "PROD-01",
        "entity_type": "well",
        "variable": "oil_rate",
        "time": 0,
    }

    assert metadata[2] == {
        "entity": "PROD-01",
        "entity_type": "well",
        "variable": "oil_rate",
        "time": 60,
    }

    assert metadata[3] == {
        "entity": "FIELD",
        "entity_type": "field",
        "variable": "water_rate",
        "time": 0,
    }


def test_build_production_observations_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="production_data cannot be empty",
    ):
        build_production_observations([])


def test_build_production_observations_requires_production_data():

    oil = ProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30],
        values=[1000, 950],
        std=[50, 50],
    )

    invalid_data = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(
        TypeError,
        match="must be ProductionData objects",
    ):
        build_production_observations([
            oil,
            invalid_data,
        ])

def test_valid_simulated_production_data():

    data = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    assert data.entity == "PROD-01"
    assert data.entity_type == "well"
    assert data.variable == "oil_rate"

    np.testing.assert_array_equal(
        data.time,
        np.array([0, 30, 60]),
    )

    np.testing.assert_array_equal(
        data.values,
        np.array([
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ]),
    )

    assert data.values.shape == (3, 3)


def test_simulated_lists_are_converted_to_numpy_arrays():

    data = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30],
        values=[
            [1000, 1050, 1100],
            [950, 1000, 1020],
        ],
    )

    assert isinstance(
        data.time,
        np.ndarray,
    )

    assert isinstance(
        data.values,
        np.ndarray,
    )


def test_simulated_time_must_be_1d():

    with pytest.raises(
        ValueError,
        match="time must be a 1D array",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[
                [0, 30],
                [60, 90],
            ],
            values=[
                [1000, 1050],
                [950, 1000],
                [900, 950],
                [850, 900],
            ],
        )


def test_simulated_values_must_be_2d():

    with pytest.raises(
        ValueError,
        match="values must be a 2D array",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[
                1000,
                950,
                900,
            ],
        )


def test_simulated_time_must_match_values_rows():

    with pytest.raises(
        ValueError,
        match="number of time points must match",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[
                0,
                30,
                60,
            ],
            values=[
                [1000, 1050],
                [950, 1000],
            ],
        )


def test_simulated_production_data_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="SimulatedProductionData cannot be empty",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[],
            values=np.empty(
                (0, 2)
            ),
        )


def test_simulated_ensemble_must_have_at_least_two_realizations():

    with pytest.raises(
        ValueError,
        match="at least 2 realizations",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[
                [1000],
                [950],
                [900],
            ],
        )


def test_simulated_values_cannot_contain_nan():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30],
            values=[
                [1000, np.nan],
                [950, 1000],
            ],
        )


def test_simulated_values_cannot_contain_infinity():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        SimulatedProductionData(
            entity="FIELD",
            entity_type="field",
            variable="water_rate",
            time=[0, 30],
            values=[
                [500, 550],
                [600, np.inf],
            ],
        )


def build_observations_and_simulations():
    """
    Build a small production problem for testing
    observation-to-simulation matching.
    """

    oil_obs = ProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[1000, 950, 900],
        std=[50, 50, 50],
    )

    water_obs = ProductionData(
        entity="FIELD",
        entity_type="field",
        variable="water_rate",
        time=[0, 30],
        values=[500, 600],
        std=[25, 25],
    )

    d_obs, Ce, metadata = (
        build_production_observations([
            oil_obs,
            water_obs,
        ])
    )

    oil_sim = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    water_sim = SimulatedProductionData(
        entity="FIELD",
        entity_type="field",
        variable="water_rate",
        time=[0, 30],
        values=[
            [480, 510, 530],
            [570, 610, 640],
        ],
    )

    return (
        d_obs,
        Ce,
        metadata,
        oil_sim,
        water_sim,
    )


def test_build_production_ensemble():

    (
        d_obs,
        Ce,
        metadata,
        oil_sim,
        water_sim,
    ) = build_observations_and_simulations()

    D = build_production_ensemble(
        simulated_data=[
            oil_sim,
            water_sim,
        ],
        metadata=metadata,
    )

    expected_D = np.array([
        [980, 1020, 1050],
        [920, 960, 990],
        [850, 910, 940],
        [480, 510, 530],
        [570, 610, 640],
    ])

    np.testing.assert_array_equal(
        D,
        expected_D,
    )

    assert D.shape == (
        len(d_obs),
        3,
    )

    assert Ce.shape == (
        len(d_obs),
        len(d_obs),
    )


def test_production_ensemble_follows_metadata_order():

    (
        _,
        _,
        metadata,
        oil_sim,
        water_sim,
    ) = build_observations_and_simulations()

    # Deliberately reverse the simulated-data order.
    #
    # D should still follow the observation metadata,
    # not the order of simulated_data.

    D = build_production_ensemble(
        simulated_data=[
            water_sim,
            oil_sim,
        ],
        metadata=metadata,
    )

    expected_D = np.array([
        [980, 1020, 1050],
        [920, 960, 990],
        [850, 910, 940],
        [480, 510, 530],
        [570, 610, 640],
    ])

    np.testing.assert_array_equal(
        D,
        expected_D,
    )


def test_production_ensemble_requires_exact_time_match():

    (
        _,
        _,
        metadata,
        _,
        water_sim,
    ) = build_observations_and_simulations()

    # Observation requires times [0, 30, 60],
    # but the simulation has [0, 30, 61].

    oil_sim = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 61],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    with pytest.raises(
        ValueError,
        match="Observation time not found",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_requires_matching_entity():

    (
        _,
        _,
        metadata,
        _,
        water_sim,
    ) = build_observations_and_simulations()

    oil_sim = SimulatedProductionData(
        entity="PROD-02",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 60],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    with pytest.raises(
        ValueError,
        match="No simulated production data found",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_requires_matching_variable():

    (
        _,
        _,
        metadata,
        _,
        water_sim,
    ) = build_observations_and_simulations()

    oil_sim = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="gas_rate",
        time=[0, 30, 60],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    with pytest.raises(
        ValueError,
        match="No simulated production data found",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_requires_same_ensemble_size():

    (
        _,
        _,
        metadata,
        oil_sim,
        _,
    ) = build_observations_and_simulations()

    water_sim = SimulatedProductionData(
        entity="FIELD",
        entity_type="field",
        variable="water_rate",
        time=[0, 30],
        values=[
            [480, 510],
            [570, 610],
        ],
    )

    with pytest.raises(
        ValueError,
        match="same number of ensemble realizations",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_cannot_have_duplicate_dataset():

    (
        _,
        _,
        metadata,
        oil_sim,
        water_sim,
    ) = build_observations_and_simulations()

    oil_sim_duplicate = (
        SimulatedProductionData(
            entity="PROD-01",
            entity_type="well",
            variable="oil_rate",
            time=[0, 30, 60],
            values=[
                [990, 1010, 1040],
                [930, 950, 980],
                [860, 900, 930],
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Multiple simulated production datasets",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                oil_sim_duplicate,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_cannot_have_duplicate_times():

    (
        _,
        _,
        metadata,
        _,
        water_sim,
    ) = build_observations_and_simulations()

    oil_sim = SimulatedProductionData(
        entity="PROD-01",
        entity_type="well",
        variable="oil_rate",
        time=[0, 30, 30],
        values=[
            [980, 1020, 1050],
            [920, 960, 990],
            [850, 910, 940],
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate time points",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=metadata,
        )


def test_production_ensemble_cannot_be_empty():

    (
        _,
        _,
        metadata,
        _,
        _,
    ) = build_observations_and_simulations()

    with pytest.raises(
        ValueError,
        match="simulated_data cannot be empty",
    ):
        build_production_ensemble(
            simulated_data=[],
            metadata=metadata,
        )


def test_production_ensemble_metadata_cannot_be_empty():

    (
        _,
        _,
        _,
        oil_sim,
        water_sim,
    ) = build_observations_and_simulations()

    with pytest.raises(
        ValueError,
        match="metadata cannot be empty",
    ):
        build_production_ensemble(
            simulated_data=[
                oil_sim,
                water_sim,
            ],
            metadata=[],
        )


def test_production_ensemble_requires_simulated_production_data():

    (
        _,
        _,
        metadata,
        _,
        water_sim,
    ) = build_observations_and_simulations()

    invalid_data = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(
        TypeError,
        match="must be SimulatedProductionData objects",
    ):
        build_production_ensemble(
            simulated_data=[
                invalid_data,
                water_sim,
            ],
            metadata=metadata,
        )