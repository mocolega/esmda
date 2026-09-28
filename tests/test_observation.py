import numpy as np
import pytest
import json

from esmda4d.esmda import (
    esmda_assimilate,
)

from esmda4d.observation import (
    DerivedObservation,
    build_observations,
    read_derived_observation,
)

from esmda4d.production import (
    read_production_data,
    build_production_observations,
)

def test_derived_observation():
    observation = DerivedObservation(
        values=np.array([
            1.0,
            2.0,
            3.0,
        ]),
        standard_deviations=np.array([
            0.1,
            0.2,
            0.3,
        ]),
    )

    np.testing.assert_allclose(
        observation.values,
        [1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(
        observation.standard_deviations,
        [0.1, 0.2, 0.3],
    )


def test_derived_observation_converts_lists():
    observation = DerivedObservation(
        values=[
            1.0,
            2.0,
        ],
        standard_deviations=[
            0.1,
            0.2,
        ],
    )

    assert isinstance(
        observation.values,
        np.ndarray,
    )

    assert isinstance(
        observation.standard_deviations,
        np.ndarray,
    )


def test_derived_observation_rejects_non_1d_values():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        DerivedObservation(
            values=np.array([
                [1.0, 2.0],
            ]),
            standard_deviations=np.array([
                0.1,
                0.2,
            ]),
        )


def test_derived_observation_rejects_non_1d_std():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        DerivedObservation(
            values=np.array([
                1.0,
                2.0,
            ]),
            standard_deviations=np.array([
                [0.1, 0.2],
            ]),
        )


def test_derived_observation_rejects_different_shapes():
    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        DerivedObservation(
            values=np.array([
                1.0,
                2.0,
            ]),
            standard_deviations=np.array([
                0.1,
            ]),
        )


def test_derived_observation_rejects_nonfinite_values():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        DerivedObservation(
            values=np.array([
                1.0,
                np.nan,
            ]),
            standard_deviations=np.array([
                0.1,
                0.2,
            ]),
        )


def test_derived_observation_rejects_nonfinite_std():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        DerivedObservation(
            values=np.array([
                1.0,
                2.0,
            ]),
            standard_deviations=np.array([
                0.1,
                np.inf,
            ]),
        )


def test_derived_observation_rejects_nonpositive_std():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        DerivedObservation(
            values=np.array([
                1.0,
                2.0,
            ]),
            standard_deviations=np.array([
                0.1,
                0.0,
            ]),
        )

def test_build_observations():
    production_d_obs = np.array([
        100.0,
        110.0,
    ])

    production_Ce = np.diag([
        4.0,
        9.0,
    ])

    derived = DerivedObservation(
        values=np.array([
            1155.0,
            1255.0,
        ]),
        standard_deviations=np.array([
            10.0,
            20.0,
        ]),
    )

    d_obs, Ce = build_observations(
        production_d_obs=production_d_obs,
        production_Ce=production_Ce,
        derived_observation=derived,
    )

    expected_d_obs = np.array([
        100.0,
        110.0,
        1155.0,
        1255.0,
    ])

    expected_Ce = np.diag([
        4.0,
        9.0,
        100.0,
        400.0,
    ])

    np.testing.assert_allclose(
        d_obs,
        expected_d_obs,
    )

    np.testing.assert_allclose(
        Ce,
        expected_Ce,
    )


def test_build_observations_without_derived_data():
    production_d_obs = np.array([
        100.0,
        110.0,
    ])

    production_Ce = np.diag([
        4.0,
        9.0,
    ])

    d_obs, Ce = build_observations(
        production_d_obs=production_d_obs,
        production_Ce=production_Ce,
    )

    np.testing.assert_allclose(
        d_obs,
        production_d_obs,
    )

    np.testing.assert_allclose(
        Ce,
        production_Ce,
    )


def test_build_observations_rejects_non_1d_production():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        build_observations(
            production_d_obs=np.array([
                [100.0, 110.0],
            ]),
            production_Ce=np.eye(2),
        )


def test_build_observations_rejects_wrong_covariance_shape():
    with pytest.raises(
        ValueError,
        match="incompatible shape",
    ):
        build_observations(
            production_d_obs=np.array([
                100.0,
                110.0,
            ]),
            production_Ce=np.eye(3),
        )


def test_build_observations_rejects_wrong_derived_type():
    with pytest.raises(
        TypeError,
        match="DerivedObservation",
    ):
        build_observations(
            production_d_obs=np.array([
                100.0,
                110.0,
            ]),
            production_Ce=np.eye(2),
            derived_observation=np.array([
                1.0,
                2.0,
            ]),
        )

def test_derived_observations_work_with_esmda():
    rng = np.random.default_rng(
        12345
    )

    M = np.array([
        [0.4, 0.5, 0.6, 0.7, 0.8],
    ])

    production_d_obs = np.array([
        2.0,
    ])

    production_Ce = np.array([
        [0.01],
    ])

    derived = DerivedObservation(
        values=np.array([
            3.0,
        ]),
        standard_deviations=np.array([
            0.2,
        ]),
    )

    d_obs, Ce = build_observations(
        production_d_obs=production_d_obs,
        production_Ce=production_Ce,
        derived_observation=derived,
    )

    def forward_model(M):
        parameter = M[0, :]

        production = (
            2.0 * parameter
        )

        derived_data = (
            3.0 * parameter
        )

        return np.vstack(
            (
                production,
                derived_data,
            )
        )

    M_updated, history = (
        esmda_assimilate(
            M=M,
            forward_model=forward_model,
            d_obs=d_obs,
            Ce=Ce,
            alphas=[1.0],
            rng=rng,
        )
    )

    assert M_updated.shape == M.shape

    assert len(history) == 2

    assert (
        history[-1]["mismatch"]
        < history[0]["mismatch"]
    )

def test_read_derived_observation(
    tmp_path,
):
    data = {
        "derived_data": {
            "values": [
                0.42,
                -0.15,
                0.73,
            ],
            "standard_deviations": [
                0.05,
                0.05,
                0.08,
            ],
        }
    }

    path = tmp_path / "observations.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
        )

    observation = (
        read_derived_observation(path)
    )

    assert isinstance(
        observation,
        DerivedObservation,
    )

    np.testing.assert_allclose(
        observation.values,
        [
            0.42,
            -0.15,
            0.73,
        ],
    )

    np.testing.assert_allclose(
        observation.standard_deviations,
        [
            0.05,
            0.05,
            0.08,
        ],
    )

def test_read_derived_observation_rejects_missing_derived_data(
    tmp_path,
):
    data = {
        "production_data": [],
    }

    path = tmp_path / "observations.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
        )

    with pytest.raises(
        ValueError,
        match="derived_data",
    ):
        read_derived_observation(path)


def test_read_derived_observation_rejects_missing_field(
    tmp_path,
):
    data = {
        "derived_data": {
            "values": [
                0.42,
                -0.15,
            ],
        }
    }

    path = tmp_path / "observations.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
        )

    with pytest.raises(
        ValueError,
        match="standard_deviations",
    ):
        read_derived_observation(path)

def test_combined_observations_from_json(
    tmp_path,
):
    data = {
        "production_data": [
            {
                "entity": "PROD1",
                "entity_type": "well",
                "variable": "OIL_RATE",
                "time": [
                    "2020-01-01",
                    "2020-02-01",
                ],
                "values": [
                    1000.0,
                    900.0,
                ],
                "standard_deviations": [
                    50.0,
                    45.0,
                ],
            },
        ],
        "derived_data": {
            "values": [
                0.42,
                -0.15,
            ],
            "standard_deviations": [
                0.05,
                0.08,
            ],
        },
    }

    path = tmp_path / "observations.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
        )

    production_data = (
        read_production_data(path)
    )

    d_prod, Ce_prod, metadata = (
        build_production_observations(
            production_data
        )
    )

    derived_observation = (
        read_derived_observation(path)
    )

    d_obs, Ce = build_observations(
        production_d_obs=d_prod,
        production_Ce=Ce_prod,
        derived_observation=derived_observation,
    )

    np.testing.assert_allclose(
        d_obs,
        [
            1000.0,
            900.0,
            0.42,
            -0.15,
        ],
    )

    np.testing.assert_allclose(
        Ce,
        np.diag([
            2500.0,
            2025.0,
            0.0025,
            0.0064,
        ]),
    )

    assert len(metadata) == 2