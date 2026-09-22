import numpy as np
import pytest

from esmda4d.grid import (
    select_grid_property,
    reconstruct_grid_property,
)


def test_select_grid_property():
    """
    Only cells for which both null_mask and
    property_mask are True should enter data
    assimilation.

    CMG cell ordering:
    I varies fastest, then J, then K.
    """

    # Grid:
    # NI = 2
    # NJ = 2
    # NK = 2
    # Ne = 2

    prior = np.zeros(
        (2, 2, 2, 2),
        dtype=float,
    )

    # Give each cell an easily identifiable value.
    #
    # realization 0: 0, 10, 20, ...
    # realization 1: 1, 11, 21, ...
    #
    # Values are assigned according to CMG cell_id.

    for j in range(2):
        values = (
            np.arange(8) * 10 + j
        )

        prior[:, :, :, j] = (
            values.reshape(
                (2, 2, 2),
                order="F",
            )
        )

    # CMG cell IDs for this grid:
    #
    # (0,0,0) -> 0
    # (1,0,0) -> 1
    # (0,1,0) -> 2
    # (1,1,0) -> 3
    # (0,0,1) -> 4
    # (1,0,1) -> 5
    # (0,1,1) -> 6
    # (1,1,1) -> 7

    null_mask = np.zeros(
        (2, 2, 2),
        dtype=bool,
    )

    # Valid cells: 0, 1, 2, 4, 5, 7

    null_mask[0, 0, 0] = True  # 0
    null_mask[1, 0, 0] = True  # 1
    null_mask[0, 1, 0] = True  # 2
    null_mask[0, 0, 1] = True  # 4
    null_mask[1, 0, 1] = True  # 5
    null_mask[1, 1, 1] = True  # 7

    property_mask = np.zeros(
        (2, 2, 2),
        dtype=bool,
    )

    # User selects cells: 1, 2, 3, 5

    property_mask[1, 0, 0] = True  # 1
    property_mask[0, 1, 0] = True  # 2
    property_mask[1, 1, 0] = True  # 3
    property_mask[1, 0, 1] = True  # 5

    prop = select_grid_property(
        variable="porosity",
        prior=prior,
        null_mask=null_mask,
        property_mask=property_mask,
    )

    # Intersection:
    #
    # NULL valid:     0, 1, 2, 4, 5, 7
    # property mask:     1, 2, 3,    5
    # ---------------------------------
    # assimilation:      1, 2,       5

    np.testing.assert_array_equal(
        prop.cell_ids,
        [1, 2, 5],
    )

    assert prop.variable == "porosity"

    assert prop.values.shape == (
        3,
        2,
    )

    # Expected:
    #
    # cell 1 -> [10, 11]
    # cell 2 -> [20, 21]
    # cell 5 -> [50, 51]

    np.testing.assert_array_equal(
        prop.values,
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [50.0, 51.0],
        ],
    )


def test_null_mask_excludes_selected_cells():
    """
    A cell selected by the property mask must not
    enter data assimilation if NULL says that the
    cell is invalid.
    """

    prior = np.ones(
        (2, 2, 2, 2)
    )

    null_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    property_mask = np.zeros(
        (2, 2, 2),
        dtype=bool,
    )

    # Cell ID 3 corresponds to:
    # (i=1, j=1, k=0)

    property_mask[1, 1, 0] = True

    # Same cell is invalid according to NULL.

    null_mask[1, 1, 0] = False

    with pytest.raises(
        ValueError,
        match="No cells selected",
    ):
        select_grid_property(
            variable="porosity",
            prior=prior,
            null_mask=null_mask,
            property_mask=property_mask,
        )


def test_reconstruct_grid_property():
    """
    Updated assimilation cells should replace the
    corresponding prior values.

    All cells outside the assimilation region must
    retain their prior values.
    """

    prior = np.zeros(
        (2, 2, 2, 2),
        dtype=float,
    )

    for j in range(2):
        values = (
            np.arange(8) * 10 + j
        )

        prior[:, :, :, j] = (
            values.reshape(
                (2, 2, 2),
                order="F",
            )
        )

    null_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    property_mask = np.zeros(
        (2, 2, 2),
        dtype=bool,
    )

    # Assimilate cells 1, 3 and 6.
    #
    # cell 1 -> (1,0,0)
    # cell 3 -> (1,1,0)
    # cell 6 -> (0,1,1)

    property_mask[1, 0, 0] = True
    property_mask[1, 1, 0] = True
    property_mask[0, 1, 1] = True

    prop = select_grid_property(
        variable="porosity",
        prior=prior,
        null_mask=null_mask,
        property_mask=property_mask,
    )

    np.testing.assert_array_equal(
        prop.cell_ids,
        [1, 3, 6],
    )

    # Simulate an ES-MDA update.

    prop.values[:] = np.array([
        [100.0, 101.0],
        [200.0, 201.0],
        [300.0, 301.0],
    ])

    reconstructed = reconstruct_grid_property(
        grid_property=prop,
        prior=prior,
    )

    # Flatten each realization using CMG ordering.

    flat_prior = np.column_stack([
        prior[:, :, :, j].ravel(
            order="F"
        )
        for j in range(2)
    ])

    flat_reconstructed = np.column_stack([
        reconstructed[:, :, :, j].ravel(
            order="F"
        )
        for j in range(2)
    ])

    # Updated cells

    np.testing.assert_array_equal(
        flat_reconstructed[1],
        [100.0, 101.0],
    )

    np.testing.assert_array_equal(
        flat_reconstructed[3],
        [200.0, 201.0],
    )

    np.testing.assert_array_equal(
        flat_reconstructed[6],
        [300.0, 301.0],
    )

    # Unchanged cells

    unchanged_cells = [
        0,
        2,
        4,
        5,
        7,
    ]

    np.testing.assert_array_equal(
        flat_reconstructed[
            unchanged_cells
        ],
        flat_prior[
            unchanged_cells
        ],
    )


def test_reconstruction_does_not_modify_prior():
    """
    Reconstruction must not modify the original
    prior array.
    """

    prior = np.arange(
        16,
        dtype=float,
    ).reshape(
        (2, 2, 2, 2)
    )

    original_prior = prior.copy()

    null_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    property_mask = np.zeros(
        (2, 2, 2),
        dtype=bool,
    )

    # CMG cell 0 = (0,0,0)

    property_mask[0, 0, 0] = True

    prop = select_grid_property(
        variable="porosity",
        prior=prior,
        null_mask=null_mask,
        property_mask=property_mask,
    )

    prop.values[:] = 999.0

    reconstruct_grid_property(
        grid_property=prop,
        prior=prior,
    )

    np.testing.assert_array_equal(
        prior,
        original_prior,
    )


def test_cmg_cell_id_ordering():
    """
    Verify CMG cell ordering using a 4 x 3 x 2 grid.

    CMG convention:
        I varies fastest,
        then J,
        then K.

    Internal cell IDs are zero-based.
    """

    NI = 4
    NJ = 3
    NK = 2

    cell_ids = np.arange(
        NI * NJ * NK
    ).reshape(
        (NI, NJ, NK),
        order="F",
    )

    # First J row, first K layer

    assert cell_ids[0, 0, 0] == 0
    assert cell_ids[1, 0, 0] == 1
    assert cell_ids[2, 0, 0] == 2
    assert cell_ids[3, 0, 0] == 3

    # Second J row, first K layer

    assert cell_ids[0, 1, 0] == 4
    assert cell_ids[1, 1, 0] == 5
    assert cell_ids[2, 1, 0] == 6
    assert cell_ids[3, 1, 0] == 7

    # Third J row, first K layer

    assert cell_ids[0, 2, 0] == 8
    assert cell_ids[3, 2, 0] == 11

    # First cell of second K layer

    assert cell_ids[0, 0, 1] == 12

    # Some additional checks

    assert cell_ids[1, 0, 1] == 13
    assert cell_ids[0, 1, 1] == 16

    # Last cell

    assert cell_ids[3, 2, 1] == 23

    # Flattening in Fortran order must reproduce
    # the CMG sequence.

    np.testing.assert_array_equal(
        cell_ids.ravel(order="F"),
        np.arange(24),
    )


def test_rejects_wrong_null_mask_shape():

    prior = np.ones(
        (2, 2, 2, 2)
    )

    null_mask = np.ones(
        (2, 2),
        dtype=bool,
    )

    property_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="null_mask",
    ):
        select_grid_property(
            variable="porosity",
            prior=prior,
            null_mask=null_mask,
            property_mask=property_mask,
        )


def test_rejects_wrong_property_mask_shape():

    prior = np.ones(
        (2, 2, 2, 2)
    )

    null_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    property_mask = np.ones(
        (2, 2),
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="property_mask",
    ):
        select_grid_property(
            variable="porosity",
            prior=prior,
            null_mask=null_mask,
            property_mask=property_mask,
        )


def test_rejects_wrong_prior_dimensions():

    prior = np.ones(
        (2, 2, 2)
    )

    null_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    property_mask = np.ones(
        (2, 2, 2),
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="prior must have shape",
    ):
        select_grid_property(
            variable="porosity",
            prior=prior,
            null_mask=null_mask,
            property_mask=property_mask,
        )