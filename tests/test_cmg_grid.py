import numpy as np
import pytest

from esmda4d.grid import select_grid_property

from esmda4d.cmg.grid import (
    load_grid_properties,
    load_grid_property,
    read_binary_grid_include,
    read_grid_include,
    read_prior_ensemble,
)


def test_read_grid_include_plain_values(tmp_path):
    path = tmp_path / "property.inc"

    path.write_text(
        "1 2 3 4\n"
        "5 6 7 8\n"
    )

    grid = read_grid_include(
        path=path,
        ni=2,
        nj=2,
        nk=2,
    )

    expected = np.arange(
        1.0,
        9.0,
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    np.testing.assert_array_equal(
        grid,
        expected,
    )


def test_read_grid_include_repeated_values(tmp_path):
    path = tmp_path / "property.inc"

    path.write_text(
        "2*1.0 3*2.0 3*4.0\n"
    )

    grid = read_grid_include(
        path=path,
        ni=2,
        nj=2,
        nk=2,
    )

    expected = np.array(
        [
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            4.0,
            4.0,
            4.0,
        ]
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    np.testing.assert_array_equal(
        grid,
        expected,
    )


def test_read_grid_include_ignores_comments(tmp_path):
    path = tmp_path / "property.inc"

    path.write_text(
        "** full-line CMG comment\n"
        "1 2 ** inline comment\n"
        "2*3 ** another comment\n"
        "4 5 6 7\n"
    )

    grid = read_grid_include(
        path=path,
        ni=2,
        nj=2,
        nk=2,
    )

    expected = np.array(
        [
            1.0,
            2.0,
            3.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ]
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    np.testing.assert_array_equal(
        grid,
        expected,
    )


def test_read_grid_include_rejects_wrong_number_of_values(
    tmp_path,
):
    path = tmp_path / "property.inc"

    path.write_text(
        "1 2 3 4 5 6 7\n"
    )

    with pytest.raises(
        ValueError,
        match="8 were expected",
    ):
        read_grid_include(
            path=path,
            ni=2,
            nj=2,
            nk=2,
        )


def test_read_grid_include_rejects_invalid_token(tmp_path):
    path = tmp_path / "property.inc"

    path.write_text(
        "1 2 3 ABC 5 6 7 8\n"
    )

    with pytest.raises(
        ValueError,
        match="Invalid numeric value",
    ):
        read_grid_include(
            path=path,
            ni=2,
            nj=2,
            nk=2,
        )


def test_read_grid_include_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.inc"

    with pytest.raises(FileNotFoundError):
        read_grid_include(
            path=path,
            ni=2,
            nj=2,
            nk=2,
        )

def test_read_binary_grid_include(tmp_path):
    path = tmp_path / "null.inc"

    path.write_text(
        "2*1 2*0\n"
        "1 0 1 0\n"
    )

    mask = read_binary_grid_include(
        path=path,
        ni=2,
        nj=2,
        nk=2,
    )

    expected = np.array(
        [
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
        ]
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    assert mask.dtype == bool

    np.testing.assert_array_equal(
        mask,
        expected,
    )


def test_read_binary_grid_include_rejects_nonbinary_values(
    tmp_path,
):
    path = tmp_path / "mask.inc"

    path.write_text(
        "1 1 0 0 1 0.5 1 0\n"
    )

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        read_binary_grid_include(
            path=path,
            ni=2,
            nj=2,
            nk=2,
        )


def test_read_prior_ensemble(tmp_path):
    paths = []

    for realization in range(3):
        path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        start = realization * 8 + 1

        values = np.arange(
            start,
            start + 8,
            dtype=float,
        )

        path.write_text(
            " ".join(
                str(value)
                for value in values
            )
        )

        paths.append(path)

    prior = read_prior_ensemble(
        paths=paths,
        ni=2,
        nj=2,
        nk=2,
        n_realizations=3,
    )

    assert prior.shape == (
        2,
        2,
        2,
        3,
    )

    for realization in range(3):
        expected = np.arange(
            realization * 8 + 1,
            realization * 8 + 9,
            dtype=float,
        ).reshape(
            (2, 2, 2),
            order="F",
        )

        np.testing.assert_array_equal(
            prior[:, :, :, realization],
            expected,
        )


def test_read_prior_ensemble_rejects_wrong_number_of_files(
    tmp_path,
):
    paths = []

    for realization in range(2):
        path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        path.write_text(
            "8*0.2\n"
        )

        paths.append(path)

    with pytest.raises(
        ValueError,
        match="Expected 3 prior files",
    ):
        read_prior_ensemble(
            paths=paths,
            ni=2,
            nj=2,
            nk=2,
            n_realizations=3,
        )

def test_cmg_grid_inputs_build_grid_property(tmp_path):
    null_path = tmp_path / "null.inc"
    mask_path = tmp_path / "por_mask.inc"

    null_path.write_text(
        "1 1 1 0 1 1 0 1\n"
    )

    mask_path.write_text(
        "1 0 1 1 1 0 1 1\n"
    )

    prior_paths = []

    for realization in range(3):
        path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        values = (
            np.arange(
                1,
                9,
                dtype=float,
            )
            + realization * 10
        )

        path.write_text(
            " ".join(
                str(value)
                for value in values
            )
        )

        prior_paths.append(path)

    null_mask = read_binary_grid_include(
        path=null_path,
        ni=2,
        nj=2,
        nk=2,
    )

    property_mask = read_binary_grid_include(
        path=mask_path,
        ni=2,
        nj=2,
        nk=2,
    )

    prior = read_prior_ensemble(
        paths=prior_paths,
        ni=2,
        nj=2,
        nk=2,
        n_realizations=3,
    )

    grid_property = select_grid_property(
        variable="POR",
        prior=prior,
        null_mask=null_mask,
        property_mask=property_mask,
    )

    # Effective mask:
    #
    # null:     1 1 1 0 1 1 0 1
    # property: 1 0 1 1 1 0 1 1
    # --------------------------------
    # effective:1 0 1 0 1 0 0 1
    #
    # Therefore cells 0, 2, 4, 7 participate
    # in history matching.

    np.testing.assert_array_equal(
        grid_property.cell_ids,
        np.array([0, 2, 4, 7]),
    )

    expected_values = np.array(
        [
            [1.0, 11.0, 21.0],
            [3.0, 13.0, 23.0],
            [5.0, 15.0, 25.0],
            [8.0, 18.0, 28.0],
        ]
    )

    np.testing.assert_array_equal(
        grid_property.values,
        expected_values,
    )

    assert grid_property.variable == "POR"

    # The complete prior is still available for
    # reconstruction after ES-MDA.
    assert prior.shape == (
        2,
        2,
        2,
        3,
    )

def test_load_grid_property_with_property_mask(tmp_path):
    null_path = tmp_path / "null.inc"
    mask_path = tmp_path / "por_mask.inc"

    null_path.write_text(
        "1 1 1 0 1 1 0 1\n"
    )

    mask_path.write_text(
        "1 0 1 1 1 0 1 1\n"
    )

    prior_paths = []

    for realization in range(3):
        path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        values = (
            np.arange(
                1,
                9,
                dtype=float,
            )
            + realization * 10
        )

        path.write_text(
            " ".join(
                str(value)
                for value in values
            )
        )

        prior_paths.append(path)

    null_mask = read_binary_grid_include(
        path=null_path,
        ni=2,
        nj=2,
        nk=2,
    )

    grid_property, prior = load_grid_property(
        variable="POR",
        prior_paths=prior_paths,
        null_mask=null_mask,
        ni=2,
        nj=2,
        nk=2,
        n_realizations=3,
        property_mask_path=mask_path,
    )

    np.testing.assert_array_equal(
        grid_property.cell_ids,
        np.array([0, 2, 4, 7]),
    )

    np.testing.assert_array_equal(
        grid_property.values,
        np.array(
            [
                [1.0, 11.0, 21.0],
                [3.0, 13.0, 23.0],
                [5.0, 15.0, 25.0],
                [8.0, 18.0, 28.0],
            ]
        ),
    )

    assert grid_property.variable == "POR"

    assert prior.shape == (
        2,
        2,
        2,
        3,
    )


def test_load_grid_property_without_property_mask(tmp_path):
    null_path = tmp_path / "null.inc"

    null_path.write_text(
        "1 1 0 1 0 1 1 0\n"
    )

    prior_paths = []

    for realization in range(3):
        path = (
            tmp_path
            / f"perm_{realization + 1:03d}.inc"
        )

        values = (
            np.arange(
                1,
                9,
                dtype=float,
            )
            + realization * 10
        )

        path.write_text(
            " ".join(
                str(value)
                for value in values
            )
        )

        prior_paths.append(path)

    null_mask = read_binary_grid_include(
        path=null_path,
        ni=2,
        nj=2,
        nk=2,
    )

    grid_property, prior = load_grid_property(
        variable="PERM",
        prior_paths=prior_paths,
        null_mask=null_mask,
        ni=2,
        nj=2,
        nk=2,
        n_realizations=3,
        property_mask_path=None,
    )

    # Without a property-specific mask, all cells
    # allowed by null_mask participate in ES-MDA.
    np.testing.assert_array_equal(
        grid_property.cell_ids,
        np.array([0, 1, 3, 5, 6]),
    )

    np.testing.assert_array_equal(
        grid_property.values,
        np.array(
            [
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [4.0, 14.0, 24.0],
                [6.0, 16.0, 26.0],
                [7.0, 17.0, 27.0],
            ]
        ),
    )

    assert grid_property.variable == "PERM"

    assert prior.shape == (
        2,
        2,
        2,
        3,
    )


def test_load_grid_properties(tmp_path):
    null_path = tmp_path / "null.inc"
    perm_mask_path = tmp_path / "perm_mask.inc"

    null_path.write_text(
        "1 1 0 1 1 1 0 1\n"
    )

    perm_mask_path.write_text(
        "1 0 1 1 0 1 1 1\n"
    )

    por_paths = []
    perm_paths = []

    for realization in range(3):
        por_path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        por_values = (
            np.arange(
                1,
                9,
                dtype=float,
            )
            + realization * 10
        )

        por_path.write_text(
            " ".join(
                str(value)
                for value in por_values
            )
        )

        por_paths.append(por_path)

        perm_path = (
            tmp_path
            / f"perm_{realization + 1:03d}.inc"
        )

        perm_values = (
            np.arange(
                101,
                109,
                dtype=float,
            )
            + realization * 100
        )

        perm_path.write_text(
            " ".join(
                str(value)
                for value in perm_values
            )
        )

        perm_paths.append(perm_path)

    properties = {
        "POR": {
            "prior_files": por_paths,
            "property_mask": None,
        },
        "PERM": {
            "prior_files": perm_paths,
            "property_mask": perm_mask_path,
        },
    }

    grid_properties, priors = load_grid_properties(
        properties=properties,
        null_file=null_path,
        ni=2,
        nj=2,
        nk=2,
        n_realizations=3,
    )

    assert len(grid_properties) == 2

    por = grid_properties[0]
    perm = grid_properties[1]

    assert por.variable == "POR"

    np.testing.assert_array_equal(
        por.cell_ids,
        np.array([0, 1, 3, 4, 5, 7]),
    )

    np.testing.assert_array_equal(
        por.values,
        np.array(
            [
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [4.0, 14.0, 24.0],
                [5.0, 15.0, 25.0],
                [6.0, 16.0, 26.0],
                [8.0, 18.0, 28.0],
            ]
        ),
    )

    assert perm.variable == "PERM"

    # null mask:
    # 1 1 0 1 1 1 0 1
    #
    # PERM mask:
    # 1 0 1 1 0 1 1 1
    #
    # effective:
    # 1 0 0 1 0 1 0 1

    np.testing.assert_array_equal(
        perm.cell_ids,
        np.array([0, 3, 5, 7]),
    )

    np.testing.assert_array_equal(
        perm.values,
        np.array(
            [
                [101.0, 201.0, 301.0],
                [104.0, 204.0, 304.0],
                [106.0, 206.0, 306.0],
                [108.0, 208.0, 308.0],
            ]
        ),
    )

    assert set(priors) == {
        "POR",
        "PERM",
    }

    assert priors["POR"].shape == (
        2,
        2,
        2,
        3,
    )

    assert priors["PERM"].shape == (
        2,
        2,
        2,
        3,
    )

    # Full priors must still contain cells excluded
    # from history matching.
    np.testing.assert_array_equal(
        priors["POR"][:, :, :, 0].reshape(
            -1,
            order="F",
        ),
        np.arange(
            1,
            9,
            dtype=float,
        ),
    )

    np.testing.assert_array_equal(
        priors["PERM"][:, :, :, 0].reshape(
            -1,
            order="F",
        ),
        np.arange(
            101,
            109,
            dtype=float,
        ),
    )