import pytest

from esmda4d.run import (
    AssimilationRun,
    exclude_realizations,
)


def test_create_assimilation_run(tmp_path):
    run = AssimilationRun(
        tmp_path / "data_assimilation_1",
        n_assimilations=4,
    )

    run.create()

    assert run.path.is_dir()
    assert run.state_path.is_dir()
    assert run.prior_path.is_dir()
    assert run.post_path.is_dir()

    for round_number in range(1, 5):
        assert run.round_path(
            round_number
        ).is_dir()


def test_round_directory_names(tmp_path):
    run = AssimilationRun(
        tmp_path / "data_assimilation_1",
        n_assimilations=4,
    )

    assert (
        run.round_path(1).name
        == "ensemble_round_001"
    )

    assert (
        run.round_path(2).name
        == "ensemble_round_002"
    )

    assert (
        run.round_path(4).name
        == "ensemble_round_004"
    )


def test_checkpoint_paths(tmp_path):
    run = AssimilationRun(
        tmp_path / "data_assimilation_1",
        n_assimilations=4,
    )

    assert (
        run.checkpoint_path(1).name
        == "round_001.npz"
    )

    assert (
        run.checkpoint_path(4).name
        == "round_004.npz"
    )

    assert (
        run.prior_checkpoint_path.name
        == "prior.npz"
    )

    assert (
        run.post_checkpoint_path.name
        == "post.npz"
    )

    assert (
        run.config_path.name
        == "config.json"
    )

    assert (
        run.observations_path.name
        == "observations.npz"
    )


def test_invalid_number_of_assimilations(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        AssimilationRun(
            tmp_path / "run",
            n_assimilations=0,
        )


def test_number_of_assimilations_must_be_integer(
    tmp_path,
):
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        AssimilationRun(
            tmp_path / "run",
            n_assimilations=4.0,
        )


@pytest.mark.parametrize(
    "round_number",
    [
        0,
        -1,
        5,
    ],
)
def test_invalid_round_number(
    tmp_path,
    round_number,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        ValueError,
        match="between 1 and 4",
    ):
        run.round_path(
            round_number
        )


def test_round_number_must_be_integer(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        TypeError,
        match="integer",
    ):
        run.round_path(1.0)


def test_create_can_be_called_twice(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "data_assimilation_1",
        n_assimilations=4,
    )

    run.create()
    run.create()

    assert run.path.is_dir()
    assert run.round_path(4).is_dir()

import numpy as np


def test_save_and_load_round_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.array([
        [0.20, 0.25],
        [100.0, 120.0],
    ])

    D = np.array([
        [300.0, 310.0],
        [50.0, 55.0],
        [10.0, 12.0],
    ])

    realization_ids = np.array([
        1,
        2,
    ])

    run.save_round_checkpoint(
        round_number=1,
        M=M,
        D=D,
        realization_ids=realization_ids,
    )

    (
        M_loaded,
        D_loaded,
        ids_loaded,
    ) = run.load_round_checkpoint(1)

    np.testing.assert_allclose(
        M_loaded,
        M,
    )

    np.testing.assert_allclose(
        D_loaded,
        D,
    )

    np.testing.assert_array_equal(
        ids_loaded,
        realization_ids,
    )


def test_checkpoint_file_exists(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.array([
        [1.0, 2.0],
    ])

    D = np.array([
        [10.0, 20.0],
    ])

    run.save_round_checkpoint(
        1,
        M,
        D,
        realization_ids=[1, 2],
    )

    assert (
        run.checkpoint_path(1)
        .is_file()
    )


def test_temporary_checkpoint_removed(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.array([
        [1.0, 2.0],
    ])

    D = np.array([
        [10.0, 20.0],
    ])

    run.save_round_checkpoint(
        1,
        M,
        D,
        realization_ids=[1, 2],
    )

    temporary = (
        run.checkpoint_path(1)
        .with_suffix(".tmp.npz")
    )

    assert not temporary.exists()


def test_missing_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        run.load_round_checkpoint(1)


def test_checkpoint_requires_matching_ensemble_size(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((3, 2))
    D = np.zeros((4, 3))

    with pytest.raises(
        ValueError,
        match="same ensemble size",
    ):
        run.save_round_checkpoint(
            1,
            M,
            D,
            realization_ids=[1, 2],
        )


def test_completed_rounds(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((2, 3))
    D = np.zeros((4, 3))
    realization_ids = [1, 2, 3]
    run.save_round_checkpoint(
        1,
        M,
        D,
        realization_ids,
    )

    run.save_round_checkpoint(
        2,
        M,
        D,
        realization_ids,
    )

    assert run.completed_rounds() == [
        1,
        2,
    ]

    assert (
        run.last_completed_round()
        == 2
    )


def test_no_completed_rounds(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    assert run.completed_rounds() == []

    assert (
        run.last_completed_round()
        is None
    )


def test_completed_rounds_stops_at_gap(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((2, 3))
    D = np.zeros((4, 3))

    realization_ids = [1, 2, 3] 

    run.save_round_checkpoint(
        1,
        M,
        D,
        realization_ids,
    )

    # Round 2 intentionally missing.

    run.save_round_checkpoint(
        3,
        M,
        D,
        realization_ids,
    )

    assert run.completed_rounds() == [1]

    assert (
        run.last_completed_round()
        == 1
    )

def test_checkpoint_preserves_nonconsecutive_ids(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    # Original realization 3 has been excluded.
    realization_ids = np.array([
        1,
        2,
        4,
        5,
    ])

    M = np.array([
        [10.0, 20.0, 40.0, 50.0],
        [11.0, 21.0, 41.0, 51.0],
    ])

    D = np.array([
        [100.0, 200.0, 400.0, 500.0],
    ])

    run.save_round_checkpoint(
        1,
        M,
        D,
        realization_ids,
    )

    (
        M_loaded,
        D_loaded,
        ids_loaded,
    ) = run.load_round_checkpoint(1)

    np.testing.assert_array_equal(
        ids_loaded,
        [1, 2, 4, 5],
    )

    # Column 2 still means original realization 4.
    assert ids_loaded[2] == 4

    np.testing.assert_allclose(
        M_loaded[:, 2],
        [40.0, 41.0],
    )

    np.testing.assert_allclose(
        D_loaded[:, 2],
        [400.0],
    )


def test_checkpoint_rejects_wrong_number_of_ids(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((2, 3))
    D = np.zeros((4, 3))

    with pytest.raises(
        ValueError,
        match="one ID",
    ):
        run.save_round_checkpoint(
            1,
            M,
            D,
            realization_ids=[
                1,
                2,
            ],
        )


def test_checkpoint_rejects_duplicate_ids(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((2, 3))
    D = np.zeros((4, 3))

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        run.save_round_checkpoint(
            1,
            M,
            D,
            realization_ids=[
                1,
                2,
                2,
            ],
        )


def test_checkpoint_rejects_nonpositive_ids(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    run.create()

    M = np.zeros((2, 3))
    D = np.zeros((4, 3))

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        run.save_round_checkpoint(
            1,
            M,
            D,
            realization_ids=[
                0,
                1,
                2,
            ],
        )

def test_exclude_realization_consistently():
    M = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
    ])

    realization_ids = np.array([
        1,
        3,
        7,
        12,
    ])

    porosity = np.zeros(
        (2, 2, 1, 4),
        dtype=float,
    )

    permeability = np.zeros(
        (2, 2, 1, 4),
        dtype=float,
    )

    # Make each realization easy to identify.
    for j in range(4):
        porosity[..., j] = (
            100.0 + j
        )

        permeability[..., j] = (
            1000.0 + j
        )

    priors = {
        "porosity": porosity,
        "permeability": permeability,
    }

    (
        M_surviving,
        priors_surviving,
        ids_surviving,
    ) = exclude_realizations(
        M=M,
        priors=priors,
        realization_ids=realization_ids,
        failed_indices=[1],
    )

    np.testing.assert_allclose(
        M_surviving,
        M[:, [0, 2, 3]],
    )

    np.testing.assert_array_equal(
        ids_surviving,
        [1, 7, 12],
    )

    np.testing.assert_allclose(
        priors_surviving["porosity"],
        porosity[..., [0, 2, 3]],
    )

    np.testing.assert_allclose(
        priors_surviving["permeability"],
        permeability[..., [0, 2, 3]],
    )

def test_exclude_multiple_realizations():
    M = np.array([
        [10.0, 20.0, 30.0, 40.0],
    ])

    priors = {
        "POR": np.array([
            [
                [
                    [
                        0.10,
                        0.20,
                        0.30,
                        0.40,
                    ]
                ]
            ]
        ])
    }

    (
        M_surviving,
        priors_surviving,
        ids_surviving,
    ) = exclude_realizations(
        M=M,
        priors=priors,
        realization_ids=[
            1,
            3,
            7,
            12,
        ],
        failed_indices=[
            1,
            3,
        ],
    )

    np.testing.assert_allclose(
        M_surviving,
        [[10.0, 30.0]],
    )

    np.testing.assert_array_equal(
        ids_surviving,
        [1, 7],
    )

    np.testing.assert_allclose(
        priors_surviving["POR"],
        priors["POR"][..., [0, 2]],
    )

def test_exclude_rejects_out_of_range_index():
    M = np.zeros((2, 3))

    priors = {
        "POR": np.zeros((1, 1, 1, 3)),
    }

    with pytest.raises(
        ValueError,
        match="outside the ensemble",
    ):
        exclude_realizations(
            M=M,
            priors=priors,
            realization_ids=[1, 2, 3],
            failed_indices=[3],
        )


def test_exclude_rejects_duplicate_indices():
    M = np.zeros((2, 3))

    priors = {
        "POR": np.zeros((1, 1, 1, 3)),
    }

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        exclude_realizations(
            M=M,
            priors=priors,
            realization_ids=[1, 2, 3],
            failed_indices=[1, 1],
        )


def test_exclude_rejects_wrong_prior_size():
    M = np.zeros((2, 3))

    priors = {
        "POR": np.zeros((1, 1, 1, 4)),
    }

    with pytest.raises(
        ValueError,
        match="does not match M",
    ):
        exclude_realizations(
            M=M,
            priors=priors,
            realization_ids=[1, 2, 3],
            failed_indices=[1],
        )