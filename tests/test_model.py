import numpy as np
import pytest

from esmda4d.model import (
    GridProperty,
    build_model_ensemble,
    unpack_model_ensemble,
)


def test_valid_grid_property():

    data = GridProperty(
        variable="permeability",
        cell_ids=[101, 102, 103],
        values=[
            [100, 120, 90, 110],
            [250, 270, 220, 260],
            [500, 450, 520, 480],
        ],
    )

    assert data.variable == "permeability"

    np.testing.assert_array_equal(
        data.cell_ids,
        np.array([101, 102, 103]),
    )

    np.testing.assert_array_equal(
        data.values,
        np.array([
            [100, 120, 90, 110],
            [250, 270, 220, 260],
            [500, 450, 520, 480],
        ]),
    )

    assert data.values.shape == (3, 4)


def test_grid_property_lists_are_converted_to_numpy_arrays():

    data = GridProperty(
        variable="porosity",
        cell_ids=[101, 102],
        values=[
            [0.20, 0.21, 0.22],
            [0.25, 0.24, 0.26],
        ],
    )

    assert isinstance(
        data.cell_ids,
        np.ndarray,
    )

    assert isinstance(
        data.values,
        np.ndarray,
    )


def test_grid_property_cell_ids_must_be_1d():

    with pytest.raises(
        ValueError,
        match="cell_ids must be a 1D array",
    ):
        GridProperty(
            variable="permeability",
            cell_ids=[
                [101, 102],
                [103, 104],
            ],
            values=[
                [100, 110],
                [200, 210],
            ],
        )


def test_grid_property_values_must_be_2d():

    with pytest.raises(
        ValueError,
        match="values must be a 2D array",
    ):
        GridProperty(
            variable="permeability",
            cell_ids=[101, 102],
            values=[
                100,
                200,
            ],
        )


def test_grid_property_cell_ids_must_match_values_rows():

    with pytest.raises(
        ValueError,
        match="number of cell IDs must match",
    ):
        GridProperty(
            variable="permeability",
            cell_ids=[
                101,
                102,
                103,
            ],
            values=[
                [100, 110],
                [200, 210],
            ],
        )


def test_grid_property_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="GridProperty cannot be empty",
    ):
        GridProperty(
            variable="permeability",
            cell_ids=[],
            values=np.empty((0, 3)),
        )


def test_grid_property_requires_at_least_two_realizations():

    with pytest.raises(
        ValueError,
        match="at least 2 realizations",
    ):
        GridProperty(
            variable="permeability",
            cell_ids=[101, 102],
            values=[
                [100],
                [200],
            ],
        )


def test_grid_property_values_cannot_contain_nan():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        GridProperty(
            variable="porosity",
            cell_ids=[101, 102],
            values=[
                [0.20, np.nan],
                [0.25, 0.26],
            ],
        )


def test_grid_property_values_cannot_contain_infinity():

    with pytest.raises(
        ValueError,
        match="values contains NaN or infinite values",
    ):
        GridProperty(
            variable="ntg",
            cell_ids=[101, 102],
            values=[
                [0.80, np.inf],
                [0.70, 0.75],
            ],
        )

def test_build_model_ensemble():

    permeability = GridProperty(
        variable="permeability",
        cell_ids=[101, 102],
        values=[
            [100, 110, 120],
            [200, 210, 220],
        ],
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=[101, 102],
        values=[
            [0.20, 0.21, 0.22],
            [0.25, 0.24, 0.26],
        ],
    )

    ntg = GridProperty(
        variable="ntg",
        cell_ids=[101, 102],
        values=[
            [0.80, 0.82, 0.84],
            [0.70, 0.72, 0.74],
        ],
    )

    M, metadata = build_model_ensemble([
        permeability,
        porosity,
        ntg,
    ])

    expected_M = np.array([
        [100, 110, 120],
        [200, 210, 220],
        [0.20, 0.21, 0.22],
        [0.25, 0.24, 0.26],
        [0.80, 0.82, 0.84],
        [0.70, 0.72, 0.74],
    ])

    np.testing.assert_array_equal(
        M,
        expected_M,
    )

    assert M.shape == (6, 3)

    assert metadata == [
        {
            "variable": "permeability",
            "cell_id": 101,
        },
        {
            "variable": "permeability",
            "cell_id": 102,
        },
        {
            "variable": "porosity",
            "cell_id": 101,
        },
        {
            "variable": "porosity",
            "cell_id": 102,
        },
        {
            "variable": "ntg",
            "cell_id": 101,
        },
        {
            "variable": "ntg",
            "cell_id": 102,
        },
    ]


def test_build_model_ensemble_preserves_property_order():

    porosity = GridProperty(
        variable="porosity",
        cell_ids=[101, 102],
        values=[
            [0.20, 0.21],
            [0.25, 0.26],
        ],
    )

    permeability = GridProperty(
        variable="permeability",
        cell_ids=[101, 102],
        values=[
            [100, 110],
            [200, 210],
        ],
    )

    M, metadata = build_model_ensemble([
        porosity,
        permeability,
    ])

    # The output must follow the order supplied by
    # the user.

    assert metadata[0]["variable"] == "porosity"
    assert metadata[1]["variable"] == "porosity"
    assert metadata[2]["variable"] == "permeability"
    assert metadata[3]["variable"] == "permeability"

    np.testing.assert_array_equal(
        M[:2],
        porosity.values,
    )

    np.testing.assert_array_equal(
        M[2:],
        permeability.values,
    )


def test_build_model_ensemble_requires_same_ensemble_size():

    permeability = GridProperty(
        variable="permeability",
        cell_ids=[101, 102],
        values=[
            [100, 110, 120],
            [200, 210, 220],
        ],
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=[101, 102],
        values=[
            [0.20, 0.21],
            [0.25, 0.26],
        ],
    )

    with pytest.raises(
        ValueError,
        match="same number of ensemble realizations",
    ):
        build_model_ensemble([
            permeability,
            porosity,
        ])


def test_build_model_ensemble_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="grid_properties cannot be empty",
    ):
        build_model_ensemble([])


def test_build_model_ensemble_requires_grid_property():

    permeability = GridProperty(
        variable="permeability",
        cell_ids=[101, 102],
        values=[
            [100, 110],
            [200, 210],
        ],
    )

    invalid_data = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(
        TypeError,
        match="must be GridProperty objects",
    ):
        build_model_ensemble([
            permeability,
            invalid_data,
        ])

def test_unpack_model_ensemble():

    M = np.array([
        [100, 110, 120],
        [200, 210, 220],
        [0.20, 0.21, 0.22],
        [0.25, 0.24, 0.26],
        [0.80, 0.82, 0.84],
        [0.70, 0.72, 0.74],
    ])

    metadata = [
        {
            "variable": "permeability",
            "cell_id": 101,
        },
        {
            "variable": "permeability",
            "cell_id": 102,
        },
        {
            "variable": "porosity",
            "cell_id": 101,
        },
        {
            "variable": "porosity",
            "cell_id": 102,
        },
        {
            "variable": "ntg",
            "cell_id": 101,
        },
        {
            "variable": "ntg",
            "cell_id": 102,
        },
    ]

    properties = unpack_model_ensemble(
        M=M,
        metadata=metadata,
    )

    assert len(properties) == 3

    permeability = properties[0]
    porosity = properties[1]
    ntg = properties[2]

    assert permeability.variable == "permeability"
    assert porosity.variable == "porosity"
    assert ntg.variable == "ntg"

    np.testing.assert_array_equal(
        permeability.cell_ids,
        [101, 102],
    )

    np.testing.assert_array_equal(
        permeability.values,
        M[0:2, :],
    )

    np.testing.assert_array_equal(
        porosity.values,
        M[2:4, :],
    )

    np.testing.assert_array_equal(
        ntg.values,
        M[4:6, :],
    )


def test_model_ensemble_round_trip():

    permeability = GridProperty(
        variable="permeability",
        cell_ids=[101, 102],
        values=[
            [100, 110, 120],
            [200, 210, 220],
        ],
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=[201, 202, 203],
        values=[
            [0.20, 0.21, 0.22],
            [0.25, 0.24, 0.26],
            [0.18, 0.19, 0.20],
        ],
    )

    ntg = GridProperty(
        variable="ntg",
        cell_ids=[301],
        values=[
            [0.80, 0.82, 0.84],
        ],
    )

    original = [
        permeability,
        porosity,
        ntg,
    ]

    M, metadata = build_model_ensemble(
        original
    )

    reconstructed = unpack_model_ensemble(
        M=M,
        metadata=metadata,
    )

    assert len(reconstructed) == len(original)

    for original_property, reconstructed_property in zip(
        original,
        reconstructed,
    ):

        assert (
            reconstructed_property.variable
            == original_property.variable
        )

        np.testing.assert_array_equal(
            reconstructed_property.cell_ids,
            original_property.cell_ids,
        )

        np.testing.assert_array_equal(
            reconstructed_property.values,
            original_property.values,
        )


def test_unpack_model_ensemble_preserves_metadata_order():

    M = np.array([
        [0.20, 0.21],
        [100, 110],
        [0.25, 0.26],
        [200, 210],
    ])

    metadata = [
        {
            "variable": "porosity",
            "cell_id": 101,
        },
        {
            "variable": "permeability",
            "cell_id": 101,
        },
        {
            "variable": "porosity",
            "cell_id": 102,
        },
        {
            "variable": "permeability",
            "cell_id": 102,
        },
    ]

    properties = unpack_model_ensemble(
        M=M,
        metadata=metadata,
    )

    assert properties[0].variable == "porosity"
    assert properties[1].variable == "permeability"

    np.testing.assert_array_equal(
        properties[0].cell_ids,
        [101, 102],
    )

    np.testing.assert_array_equal(
        properties[0].values,
        M[[0, 2], :],
    )

    np.testing.assert_array_equal(
        properties[1].cell_ids,
        [101, 102],
    )

    np.testing.assert_array_equal(
        properties[1].values,
        M[[1, 3], :],
    )


def test_unpack_model_ensemble_requires_2d_M():

    with pytest.raises(
        ValueError,
        match="M must be a 2D array",
    ):
        unpack_model_ensemble(
            M=[1, 2, 3],
            metadata=[
                {
                    "variable": "porosity",
                    "cell_id": 101,
                },
                {
                    "variable": "porosity",
                    "cell_id": 102,
                },
                {
                    "variable": "porosity",
                    "cell_id": 103,
                },
            ],
        )


def test_unpack_model_ensemble_requires_matching_metadata():

    M = np.array([
        [100, 110],
        [200, 210],
    ])

    metadata = [
        {
            "variable": "permeability",
            "cell_id": 101,
        },
    ]

    with pytest.raises(
        ValueError,
        match="number of rows in M must match",
    ):
        unpack_model_ensemble(
            M=M,
            metadata=metadata,
        )


def test_unpack_model_ensemble_cannot_be_empty():

    with pytest.raises(
        ValueError,
        match="M cannot be empty",
    ):
        unpack_model_ensemble(
            M=np.empty((0, 3)),
            metadata=[],
        )


def test_unpack_model_ensemble_requires_two_realizations():

    M = np.array([
        [100],
        [200],
    ])

    metadata = [
        {
            "variable": "permeability",
            "cell_id": 101,
        },
        {
            "variable": "permeability",
            "cell_id": 102,
        },
    ]

    with pytest.raises(
        ValueError,
        match="at least 2 realizations",
    ):
        unpack_model_ensemble(
            M=M,
            metadata=metadata,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
    ],
)
def test_unpack_model_ensemble_requires_finite_M(
    invalid_value,
):

    M = np.array([
        [100, 110],
        [200, invalid_value],
    ])

    metadata = [
        {
            "variable": "permeability",
            "cell_id": 101,
        },
        {
            "variable": "permeability",
            "cell_id": 102,
        },
    ]

    with pytest.raises(
        ValueError,
        match="M contains NaN or infinite values",
    ):
        unpack_model_ensemble(
            M=M,
            metadata=metadata,
        )