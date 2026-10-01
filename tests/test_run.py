import pytest

from esmda4d.run import (
    AssimilationRun,
    EnsembleEvaluation,
    exclude_realizations,
    evaluate_ensemble,
    assimilate_round,
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

    for round_number in range(1, 4):
        assert run.round_path(
            round_number
        ).is_dir()

    assert not (
        run.path / "round_004"
    ).exists()


def test_run_directory_names(tmp_path):
    run = AssimilationRun(
        tmp_path / "data_assimilation_1",
        n_assimilations=4,
    )

    assert run.prior_path.name == "prior"

    assert (
        run.round_path(1).name
        == "round_001"
    )

    assert (
        run.round_path(2).name
        == "round_002"
    )

    assert (
        run.round_path(3).name
        == "round_003"
    )

    assert run.post_path.name == "post"


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
        run.checkpoint_path(3).name
        == "round_003.npz"
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
        4,
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
        match="between 1 and 3",
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
    assert run.prior_path.is_dir()
    assert run.round_path(3).is_dir()
    assert run.post_path.is_dir()
    

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

class FakeRealizationFailure(RuntimeError):
    def __init__(
        self,
        failed_indices,
        failed_ids,
    ):
        self.failed_indices = failed_indices
        self.failed_ids = failed_ids


class FakeExcludingForwardModel:
    def __init__(self):
        self.calls = []
        self.realization_ids = None
        self.priors = None

    def set_ensemble_context(
        self,
        priors,
        realization_ids,
    ):
        self.priors = priors
        self.realization_ids = list(
            realization_ids
        )

    def __call__(self, M):
        self.calls.append(
            list(self.realization_ids)
        )

        if 3 in self.realization_ids:
            j = self.realization_ids.index(3)

            raise FakeRealizationFailure(
                failed_indices=[j],
                failed_ids=[3],
            )

        return M[:1, :] * 10.0

def test_evaluate_ensemble_excludes_failed_realization():
    M = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
    ])

    priors = {
        "POR": np.zeros(
            (2, 2, 1, 4)
        ),
    }

    forward = FakeExcludingForwardModel()

    result = evaluate_ensemble(
        M=M,
        priors=priors,
        realization_ids=[
            1,
            3,
            7,
            12,
        ],
        forward_model=forward,
        failure_exception=(
            FakeRealizationFailure
        ),
    )

    assert isinstance(
        result,
        EnsembleEvaluation,
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        [1, 7, 12],
    )

    assert result.excluded_ids == [3]

    np.testing.assert_allclose(
        result.M,
        M[:, [0, 2, 3]],
    )

    np.testing.assert_allclose(
        result.D,
        [[100.0, 300.0, 400.0]],
    )

    assert forward.calls == [
        [1, 3, 7, 12],
        [1, 7, 12],
    ]

    assert (
        result.priors["POR"].shape[-1]
        == 3
    )

def test_evaluate_ensemble_without_failure():
    M = np.array([
        [10.0, 20.0, 30.0],
    ])

    priors = {
        "POR": np.zeros(
            (1, 1, 1, 3)
        ),
    }

    forward = FakeExcludingForwardModel()

    result = evaluate_ensemble(
        M=M,
        priors=priors,
        realization_ids=[
            1,
            7,
            12,
        ],
        forward_model=forward,
        failure_exception=(
            FakeRealizationFailure
        ),
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        [1, 7, 12],
    )

    assert result.excluded_ids == []

    assert forward.calls == [
        [1, 7, 12],
    ]

def test_evaluate_ensemble_rejects_too_few_survivors():
    M = np.zeros((2, 2))

    priors = {
        "POR": np.zeros(
            (1, 1, 1, 2)
        ),
    }

    forward = FakeExcludingForwardModel()

    with pytest.raises(
        RuntimeError,
        match="Fewer than two",
    ):
        evaluate_ensemble(
            M=M,
            priors=priors,
            realization_ids=[
                1,
                3,
            ],
            forward_model=forward,
            failure_exception=(
                FakeRealizationFailure
            ),
        )

def test_save_observations(tmp_path):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    d_obs = np.array([
        1000.0,
        900.0,
        0.42,
        -0.15,
    ])

    Ce = np.diag([
        2500.0,
        2025.0,
        0.0025,
        0.0064,
    ])

    run.save_observations(
        d_obs=d_obs,
        Ce=Ce,
    )

    assert (
        run.observations_path.is_file()
    )

    with np.load(
        run.observations_path,
        allow_pickle=False,
    ) as data:

        np.testing.assert_allclose(
            data["d_obs"],
            d_obs,
        )

        np.testing.assert_allclose(
            data["Ce"],
            Ce,
        )

def test_save_observations_rejects_non_1d_d_obs(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        ValueError,
        match="1D",
    ):
        run.save_observations(
            d_obs=np.array([
                [1.0, 2.0],
            ]),
            Ce=np.eye(2),
        )


def test_save_observations_rejects_wrong_Ce_shape(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        ValueError,
        match="n_data",
    ):
        run.save_observations(
            d_obs=np.array([
                1.0,
                2.0,
            ]),
            Ce=np.eye(3),
        )


def test_save_observations_rejects_nonsymmetric_Ce(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    Ce = np.array([
        [1.0, 0.5],
        [0.0, 1.0],
    ])

    with pytest.raises(
        ValueError,
        match="symmetric",
    ):
        run.save_observations(
            d_obs=np.array([
                1.0,
                2.0,
            ]),
            Ce=Ce,
        )

def test_save_and_load_observations(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    d_obs = np.array([
        1000.0,
        900.0,
        0.42,
        -0.15,
    ])

    Ce = np.diag([
        2500.0,
        2025.0,
        0.0025,
        0.0064,
    ])

    run.save_observations(
        d_obs=d_obs,
        Ce=Ce,
    )

    loaded_d_obs, loaded_Ce = (
        run.load_observations()
    )

    np.testing.assert_allclose(
        loaded_d_obs,
        d_obs,
    )

    np.testing.assert_allclose(
        loaded_Ce,
        Ce,
    )


def test_load_observations_missing_file(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Observation checkpoint not found",
    ):
        run.load_observations()


def test_load_observations_rejects_invalid_contents(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    run.state_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        run.observations_path,
        wrong=np.array([1.0]),
    )

    with pytest.raises(
        ValueError,
        match="d_obs and Ce",
    ):
        run.load_observations()

def test_assimilate_round_excludes_failed_realization():
    M = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    priors = {
        "POR": np.zeros(
            (1, 1, 1, 4)
        ),
    }

    forward = FakeExcludingForwardModel()

    d_obs = np.array([
        8.0,
    ])

    Ce = np.array([
        [0.01],
    ])

    rng = np.random.default_rng(12345)

    evaluation, M_updated = (
        assimilate_round(
            M=M,
            priors=priors,
            realization_ids=[
                1,
                3,
                7,
                12,
            ],
            forward_model=forward,
            failure_exception=(
                FakeRealizationFailure
            ),
            d_obs=d_obs,
            Ce=Ce,
            alpha=1.0,
            rng=rng,
        )
    )

    np.testing.assert_array_equal(
        evaluation.realization_ids,
        [1, 7, 12],
    )

    assert evaluation.excluded_ids == [3]

    np.testing.assert_allclose(
        evaluation.M,
        M[:, [0, 2, 3]],
    )

    assert evaluation.D.shape == (
        1,
        3,
    )

    assert M_updated.shape == (
        1,
        3,
    )

    assert np.all(
        np.isfinite(M_updated)
    )

def test_save_prior_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.array([
        [0.10, 0.20, 0.30],
        [100.0, 200.0, 300.0],
    ])

    D = np.array([
        [10.0, 20.0, 30.0],
    ])

    realization_ids = np.array([
        1,
        2,
        3,
    ])

    porosity = np.zeros(
        (2, 2, 1, 3)
    )

    permeability = np.zeros(
        (2, 2, 1, 3)
    )

    for j in range(3):
        porosity[..., j] = (
            0.10 + 0.05 * j
        )

        permeability[..., j] = (
            100.0 + 50.0 * j
        )

    priors = {
        "POR": porosity,
        "PERMI": permeability,
    }

    run.save_prior_checkpoint(
        M=M,
        D=D,
        priors=priors,
        realization_ids=realization_ids,
    )

    assert (
        run.prior_checkpoint_path.is_file()
    )

    with np.load(
        run.prior_checkpoint_path,
        allow_pickle=False,
    ) as data:

        assert set(data.files) == {
            "M",
            "D",
            "realization_ids",
            "prior__POR",
            "prior__PERMI",
        }

        np.testing.assert_allclose(
            data["M"],
            M,
        )

        np.testing.assert_array_equal(
            data["D"],
            D,
        )

        np.testing.assert_array_equal(
            data["realization_ids"],
            realization_ids,
        )

        np.testing.assert_allclose(
            data["prior__POR"],
            porosity,
        )

        np.testing.assert_allclose(
            data["prior__PERMI"],
            permeability,
        )

def test_save_prior_checkpoint_rejects_wrong_prior_size(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.zeros(
        (2, 3)
    )

    D = np.zeros(
       (1, 3)
    )

    priors = {
        "POR": np.zeros(
            (2, 2, 1, 4)
        ),
    }

    with pytest.raises(
        ValueError,
        match="does not match M",
    ):
        run.save_prior_checkpoint(
            M=M,
            D=D,
            priors=priors,
            realization_ids=[
                1,
                2,
                3,
            ],
        )


def test_save_prior_checkpoint_rejects_duplicate_ids(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.zeros(
        (2, 3)
    )

    D = np.zeros(
       (1, 3)
    )

    priors = {
        "POR": np.zeros(
            (2, 2, 1, 3)
        ),
    }

    with pytest.raises(
            ValueError,
            match="unique",
        ):
            run.save_prior_checkpoint(
                M=M,
                D=D,
                priors=priors,
                realization_ids=[
                    1,
                    2,
                    2,
                ],
            )

def test_save_and_load_prior_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        tmp_path / "run",
        n_assimilations=4,
    )

    M = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    D = np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )

    priors = {
        "POR": np.arange(
            2 * 2 * 1 * 3,
            dtype=float,
        ).reshape(2, 2, 1, 3)
    }

    realization_ids = np.array(
        [1, 3, 5]
    )

    run.save_prior_checkpoint(
        M=M,
        D=D,
        priors=priors,
        realization_ids=realization_ids,
    )

    (
        M_loaded,
        D_loaded,
        priors_loaded,
        ids_loaded,
    ) = run.load_prior_checkpoint()

    np.testing.assert_array_equal(
        M_loaded,
        M,
    )

    np.testing.assert_array_equal(
        D_loaded,
        D,
    )

    np.testing.assert_array_equal(
        priors_loaded["POR"],
        priors["POR"],
    )

    np.testing.assert_array_equal(
        ids_loaded,
        realization_ids,
    )


def test_load_prior_checkpoint_missing_file(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Prior checkpoint not found",
    ):
        run.load_prior_checkpoint()


def test_load_prior_checkpoint_rejects_invalid_contents(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    run.state_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        run.prior_checkpoint_path,
        wrong=np.array([1.0]),
    )

    with pytest.raises(
        ValueError,
        match="M, D, and realization_ids",
    ):
        run.load_prior_checkpoint()

def test_save_post_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.array([
        [0.80, 0.90, 1.00],
        [100.0, 110.0, 120.0],
    ])

    D = np.array([
        [800.0, 900.0, 1000.0],
        [80.0, 90.0, 100.0],
    ])

    realization_ids = np.array([
        1,
        3,
        7,
    ])

    run.save_post_checkpoint(
        M=M,
        D=D,
        realization_ids=realization_ids,
    )

    assert (
        run.post_checkpoint_path.is_file()
    )

    with np.load(
        run.post_checkpoint_path,
        allow_pickle=False,
    ) as data:

        assert set(data.files) == {
            "M",
            "D",
            "realization_ids",
        }

        np.testing.assert_allclose(
            data["M"],
            M,
        )

        np.testing.assert_allclose(
            data["D"],
            D,
        )

        np.testing.assert_array_equal(
            data["realization_ids"],
            realization_ids,
        )


def test_save_post_checkpoint_rejects_mismatched_ensemble_size(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.zeros(
        (2, 3)
    )

    D = np.zeros(
        (4, 2)
    )

    with pytest.raises(
        ValueError,
        match="same ensemble size",
    ):
        run.save_post_checkpoint(
            M=M,
            D=D,
            realization_ids=[
                1,
                2,
                3,
            ],
        )


def test_save_post_checkpoint_rejects_duplicate_ids(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.zeros(
        (2, 3)
    )

    D = np.zeros(
        (4, 3)
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        run.save_post_checkpoint(
            M=M,
            D=D,
            realization_ids=[
                1,
                2,
                2,
            ],
        )

def test_save_and_load_post_checkpoint(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    M = np.array([
        [0.80, 0.90, 1.00],
        [100.0, 110.0, 120.0],
    ])

    D = np.array([
        [800.0, 900.0, 1000.0],
        [80.0, 90.0, 100.0],
    ])

    realization_ids = np.array([
        1,
        3,
        7,
    ])

    run.save_post_checkpoint(
        M=M,
        D=D,
        realization_ids=realization_ids,
    )

    (
        M_loaded,
        D_loaded,
        ids_loaded,
    ) = run.load_post_checkpoint()

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


def test_load_post_checkpoint_missing_file(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Post checkpoint not found",
    ):
        run.load_post_checkpoint()


def test_load_post_checkpoint_rejects_invalid_contents(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=4,
    )

    run.state_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        run.post_checkpoint_path,
        wrong=np.array([1.0]),
    )

    with pytest.raises(
        ValueError,
        match="M, D, and realization_ids",
    ):
        run.load_post_checkpoint()

def test_assimilation_run_assimilate_two_rounds(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=2,
    )

    M = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    priors = {
        "POR": np.zeros(
            (1, 1, 1, 4)
        ),
    }

    realization_ids = np.array([
        1,
        2,
        3,
        4,
    ])

    d_obs = np.array([
        8.0,
    ])

    Ce = np.array([
        [0.01],
    ])

    alphas = np.array([
        2.0,
        2.0,
    ])

    rng = np.random.default_rng(
        12345
    )

    class FakeForwardModel:
        def __init__(self):
            self.calls = []
            self.priors = None
            self.realization_ids = None

        def set_ensemble_context(
            self,
            priors,
            realization_ids,
        ):
            self.priors = priors
            self.realization_ids = (
                np.asarray(
                    realization_ids
                ).copy()
            )

        def __call__(self, M):
            D = M[:1, :] * 10.0

            self.calls.append({
                "M": M.copy(),
                "D": D.copy(),
                "realization_ids":
                    self.realization_ids.copy(),
            })

            return D

    forward = FakeForwardModel()

    result = run.assimilate(
        M=M,
        priors=priors,
        realization_ids=realization_ids,
        forward_model=forward,
        failure_exception=(
            FakeRealizationFailure
        ),
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng,
    )

    assert isinstance(
        result,
        EnsembleEvaluation,
    )

    # Two ES-MDA updates require three
    # forward-model evaluations:
    #
    # M0 -> D0
    # M1 -> D1
    # M2 -> D2
    assert len(forward.calls) == 3

    # Prior checkpoint contains the
    # evaluated prior ensemble M0, D0.
    (
        M_prior,
        D_prior,
        priors_prior,
        ids_prior,
    ) = run.load_prior_checkpoint()

    np.testing.assert_allclose(
        M_prior,
        forward.calls[0]["M"],
    )

    np.testing.assert_allclose(
        D_prior,
        forward.calls[0]["D"],
    )

    np.testing.assert_array_equal(
        ids_prior,
        forward.calls[0][
            "realization_ids"
        ],
    )

    np.testing.assert_allclose(
        priors_prior["POR"],
        priors["POR"],
    )

    # With two assimilations there is
    # only one intermediate checkpoint.
    # round_001 contains M1, D1.
    (
        M_round_1,
        D_round_1,
        ids_round_1,
    ) = run.load_round_checkpoint(1)

    np.testing.assert_allclose(
        M_round_1,
        forward.calls[1]["M"],
    )

    np.testing.assert_allclose(
        D_round_1,
        forward.calls[1]["D"],
    )

    np.testing.assert_array_equal(
        ids_round_1,
        forward.calls[1][
            "realization_ids"
        ],
    )

    # There is no round_002 because M2
    # is the posterior ensemble.
    assert not (
        run.state_path
        / "round_002.npz"
    ).exists()

    # Posterior checkpoint contains the
    # final evaluated ensemble M2, D2.
    (
        M_post,
        D_post,
        ids_post,
    ) = run.load_post_checkpoint()

    np.testing.assert_allclose(
        M_post,
        forward.calls[2]["M"],
    )

    np.testing.assert_allclose(
        D_post,
        forward.calls[2]["D"],
    )

    np.testing.assert_array_equal(
        ids_post,
        forward.calls[2][
            "realization_ids"
        ],
    )

    # Returned result is the same final
    # evaluated posterior.
    np.testing.assert_allclose(
        result.M,
        M_post,
    )

    np.testing.assert_allclose(
        result.D,
        D_post,
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        ids_post,
    )

    assert result.excluded_ids == []

def test_assimilation_run_propagates_excluded_realization(
    tmp_path,
):
    run = AssimilationRun(
        path=tmp_path / "run",
        n_assimilations=2,
    )

    M = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    priors = {
        "POR": np.array(
            [[[[10.0, 20.0, 30.0, 40.0]]]]
        ),
    }

    realization_ids = np.array([
        1,
        2,
        3,
        4,
    ])

    d_obs = np.array([
        8.0,
    ])

    Ce = np.array([
        [0.01],
    ])

    alphas = np.array([
        2.0,
        2.0,
    ])

    rng = np.random.default_rng(
        12345
    )

    class FakeForwardModel:
        def __init__(self):
            self.calls = []
            self.priors = None
            self.realization_ids = None
            self.has_failed = False

        def set_ensemble_context(
            self,
            priors,
            realization_ids,
        ):
            self.priors = priors
            self.realization_ids = (
                np.asarray(
                    realization_ids
                ).copy()
            )

        def __call__(self, M):
            self.calls.append(
                self.realization_ids.copy()
            )

            # Fail realization ID 2 only on
            # the first attempt.
            if (
                not self.has_failed
                and 2 in self.realization_ids
            ):
                self.has_failed = True

                failed_index = np.where(
                    self.realization_ids == 2
                )[0][0]

                raise FakeRealizationFailure(
                    failed_indices=[
                        failed_index
                    ],
                    failed_ids=[2],
                )

            return M[:1, :] * 10.0

    forward = FakeForwardModel()

    result = run.assimilate(
        M=M,
        priors=priors,
        realization_ids=realization_ids,
        forward_model=forward,
        failure_exception=(
            FakeRealizationFailure
        ),
        d_obs=d_obs,
        Ce=Ce,
        alphas=alphas,
        rng=rng,
    )

    # First prior attempt contains all
    # realizations.
    np.testing.assert_array_equal(
        forward.calls[0],
        [1, 2, 3, 4],
    )

    # Prior retry after the failure contains
    # only the surviving realizations.
    np.testing.assert_array_equal(
        forward.calls[1],
        [1, 3, 4],
    )

    # Every subsequent evaluation must keep
    # using only the surviving realizations.
    for ids in forward.calls[2:]:
        np.testing.assert_array_equal(
            ids,
            [1, 3, 4],
        )

    # The prior checkpoint is written only
    # after a successful prior evaluation.
    # Therefore it contains only survivors.
    (
        M_prior,
        D_prior,
        priors_prior,
        ids_prior,
    ) = run.load_prior_checkpoint()

    assert M_prior.shape[1] == 3
    assert D_prior.shape[1] == 3

    np.testing.assert_array_equal(
        ids_prior,
        [1, 3, 4],
    )

    np.testing.assert_allclose(
        M_prior,
        M[:, [0, 2, 3]],
    )

    np.testing.assert_allclose(
        priors_prior["POR"],
        np.array(
            [[[[10.0, 30.0, 40.0]]]]
        ),
    )

    # With two assimilations there is only
    # one intermediate checkpoint.
    (
        M_round,
        D_round,
        ids_round,
    ) = run.load_round_checkpoint(1)

    assert M_round.shape[1] == 3
    assert D_round.shape[1] == 3

    np.testing.assert_array_equal(
        ids_round,
        [1, 3, 4],
    )

    # There must be no round_002 checkpoint.
    assert not (
        run.state_path
        / "round_002.npz"
    ).exists()

    # The final posterior must also contain
    # only the surviving realizations.
    (
        M_post,
        D_post,
        ids_post,
    ) = run.load_post_checkpoint()

    assert M_post.shape[1] == 3
    assert D_post.shape[1] == 3

    np.testing.assert_array_equal(
        ids_post,
        [1, 3, 4],
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        [1, 3, 4],
    )

    # The matching prior column must remain
    # excluded throughout the run.
    np.testing.assert_allclose(
        result.priors["POR"],
        np.array(
            [[[[10.0, 30.0, 40.0]]]]
        ),
    )

    assert result.excluded_ids == []


def test_assimilation_run_resumes_after_first_round(
    tmp_path,
):
    """
    A run interrupted after round_001 should resume
    from that evaluated checkpoint, perform update #2,
    and evaluate only the final posterior.
    """

    run = AssimilationRun(
        path=tmp_path,
        n_assimilations=2,
    )

    M_prior = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    D_prior = np.array([
        [4.0, 6.0, 8.0, 10.0],
    ])

    priors = {
        "POR": np.array([
            [[
                [10.0, 20.0, 30.0, 40.0]
            ]]
        ]),
    }

    original_ids = np.array([
        1, 2, 3, 4
    ])

    run.create()

    run.save_prior_checkpoint(
        M=M_prior,
        D=D_prior,
        priors=priors,
        realization_ids=original_ids,
    )

    # Simulate a completed round_001 in which
    # realization 2 was excluded.
    M_round_1 = np.array([
        [0.4, 0.8, 1.0],
    ])

    D_round_1 = np.array([
        [4.0, 8.0, 10.0],
    ])

    surviving_ids = np.array([
        1, 3, 4
    ])

    run.save_round_checkpoint(
        round_number=1,
        M=M_round_1,
        D=D_round_1,
        realization_ids=surviving_ids,
    )

    calls = []

    class FakeForwardModel:

        def set_ensemble_context(
            self,
            priors,
            realization_ids,
        ):
            self.priors = priors
            self.realization_ids = np.asarray(
                realization_ids
            ).copy()

        def __call__(self, M):
            D = M[:1, :] * 10.0

            calls.append({
                "M": np.asarray(M).copy(),
                "D": D.copy(),
                "ids": self.realization_ids.copy(),
                "priors": {
                    name: values.copy()
                    for name, values
                    in self.priors.items()
                },
            })

            return D

    forward_model = FakeForwardModel()

    result = run.resume(
        forward_model=forward_model,
        failure_exception=FakeRealizationFailure,
        d_obs=np.array([8.0]),
        Ce=np.array([[0.01]]),
        alphas=np.array([2.0, 2.0]),
        rng=np.random.default_rng(12345),
    )

    # round_001 already contains M1, D1.
    # Resume performs update #2 and evaluates
    # only the final posterior M2, D2.
    assert len(calls) == 1

    # Realization 2 must remain excluded.
    np.testing.assert_array_equal(
        calls[0]["ids"],
        surviving_ids,
    )

    # Priors must be selected according to the
    # realization IDs surviving in round_001.
    expected_por = np.array([
        [[
            [10.0, 30.0, 40.0]
        ]]
    ])

    np.testing.assert_array_equal(
        calls[0]["priors"]["POR"],
        expected_por,
    )

    # The posterior checkpoint corresponds to
    # the only forward evaluation after resume.
    (
        M_post,
        D_post,
        ids_post,
    ) = run.load_post_checkpoint()

    np.testing.assert_allclose(
        M_post,
        calls[0]["M"],
    )

    np.testing.assert_allclose(
        D_post,
        calls[0]["D"],
    )

    np.testing.assert_array_equal(
        ids_post,
        surviving_ids,
    )

    np.testing.assert_allclose(
        result.M,
        M_post,
    )

    np.testing.assert_allclose(
        result.D,
        D_post,
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        surviving_ids,
    )


def test_assimilation_run_resumes_after_last_round(
    tmp_path,
):
    """
    If the last intermediate round is already
    checkpointed, resume should perform the final
    ES-MDA update and evaluate only the posterior.
    """

    run = AssimilationRun(
        path=tmp_path,
        n_assimilations=4,
    )

    M_prior = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    D_prior = np.array([
        [4.0, 6.0, 8.0, 10.0],
    ])

    priors = {
        "POR": np.array([
            [[
                [10.0, 20.0, 30.0, 40.0]
            ]]
        ]),
    }

    original_ids = np.array([
        1, 2, 3, 4
    ])

    run.create()

    run.save_prior_checkpoint(
        M=M_prior,
        D=D_prior,
        priors=priors,
        realization_ids=original_ids,
    )

    # Simulate completed intermediate rounds.
    #
    # Realization 2 was excluded during the run.

    surviving_ids = np.array([
        1, 3, 4
    ])

    run.save_round_checkpoint(
        round_number=1,
        M=np.array([
            [0.40, 0.80, 1.00],
        ]),
        D=np.array([
            [4.0, 8.0, 10.0],
        ]),
        realization_ids=surviving_ids,
    )

    run.save_round_checkpoint(
        round_number=2,
        M=np.array([
            [0.45, 0.78, 0.95],
        ]),
        D=np.array([
            [4.5, 7.8, 9.5],
        ]),
        realization_ids=surviving_ids,
    )

    M_round_3 = np.array([
        [0.50, 0.75, 0.90],
    ])

    D_round_3 = np.array([
        [5.0, 7.5, 9.0],
    ])

    run.save_round_checkpoint(
        round_number=3,
        M=M_round_3,
        D=D_round_3,
        realization_ids=surviving_ids,
    )

    calls = []

    class FakeForwardModel:

        def set_ensemble_context(
            self,
            priors,
            realization_ids,
        ):
            self.priors = priors
            self.realization_ids = np.asarray(
                realization_ids
            ).copy()

        def __call__(self, M):
            D = M[:1, :] * 10.0

            calls.append({
                "M": np.asarray(M).copy(),
                "D": D.copy(),
                "ids": self.realization_ids.copy(),
                "priors": {
                    name: values.copy()
                    for name, values
                    in self.priors.items()
                },
            })

            return D

    forward_model = FakeForwardModel()

    result = run.resume(
        forward_model=forward_model,
        failure_exception=FakeRealizationFailure,
        d_obs=np.array([8.0]),
        Ce=np.array([[0.01]]),
        alphas=np.array([
            4.0,
            4.0,
            4.0,
            4.0,
        ]),
        rng=np.random.default_rng(12345),
    )

    # round_003 already contains M3, D3.
    #
    # Resume must therefore perform only:
    #
    #   update #4 -> M4
    #   evaluate M4 -> D4
    #
    # Thus only one forward evaluation is needed.

    assert len(calls) == 1

    # The final update must start from the
    # realizations surviving in round_003.

    np.testing.assert_array_equal(
        calls[0]["ids"],
        surviving_ids,
    )

    # The original full-grid priors must be selected
    # according to the surviving realization IDs.

    expected_por = np.array([
        [[
            [10.0, 30.0, 40.0]
        ]]
    ])

    np.testing.assert_array_equal(
        calls[0]["priors"]["POR"],
        expected_por,
    )

    # The forward evaluation after resume is the
    # final posterior M4, D4.

    (
        M_post,
        D_post,
        ids_post,
    ) = run.load_post_checkpoint()

    np.testing.assert_allclose(
        M_post,
        calls[0]["M"],
    )

    np.testing.assert_allclose(
        D_post,
        calls[0]["D"],
    )

    np.testing.assert_array_equal(
        ids_post,
        surviving_ids,
    )

    # The returned evaluation must correspond to
    # the saved posterior.

    np.testing.assert_allclose(
        result.M,
        M_post,
    )

    np.testing.assert_allclose(
        result.D,
        D_post,
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        surviving_ids,
    )

def test_assimilation_run_resumes_from_prior(
    tmp_path,
):
    """
    If the evaluated prior checkpoint exists but no
    intermediate round was completed, resume should
    continue from M0, D0 without evaluating the prior
    again.
    """

    run = AssimilationRun(
        path=tmp_path,
        n_assimilations=2,
    )

    M_prior = np.array([
        [0.4, 0.6, 0.8, 1.0],
    ])

    D_prior = np.array([
        [4.0, 6.0, 8.0, 10.0],
    ])

    priors = {
        "POR": np.array([
            [[
                [10.0, 20.0, 30.0, 40.0]
            ]]
        ]),
    }

    realization_ids = np.array([
        1, 2, 3, 4
    ])

    run.create()

    # Simulate a run that successfully evaluated
    # the prior, saved M0 and D0, and then stopped
    # before ES-MDA update #1.
    run.save_prior_checkpoint(
        M=M_prior,
        D=D_prior,
        priors=priors,
        realization_ids=realization_ids,
    )

    calls = []

    class FakeForwardModel:

        def set_ensemble_context(
            self,
            priors,
            realization_ids,
        ):
            self.priors = priors
            self.realization_ids = np.asarray(
                realization_ids
            ).copy()

        def __call__(self, M):
            D = M[:1, :] * 10.0

            calls.append({
                "M": np.asarray(M).copy(),
                "D": D.copy(),
                "ids": self.realization_ids.copy(),
                "priors": {
                    name: values.copy()
                    for name, values
                    in self.priors.items()
                },
            })

            return D

    forward_model = FakeForwardModel()

    result = run.resume(
        forward_model=forward_model,
        failure_exception=FakeRealizationFailure,
        d_obs=np.array([8.0]),
        Ce=np.array([[0.01]]),
        alphas=np.array([2.0, 2.0]),
        rng=np.random.default_rng(12345),
    )

    # The prior M0, D0 is already available.
    # Resume must therefore perform only:
    #
    #   update #1
    #   M1 -> D1
    #   update #2
    #   M2 -> D2
    #
    # Thus only two forward evaluations are needed.
    assert len(calls) == 2

    # The first forward evaluation after resume is
    # M1, not the saved prior M0.
    assert not np.array_equal(
        calls[0]["M"],
        M_prior,
    )

    np.testing.assert_array_equal(
        calls[0]["ids"],
        realization_ids,
    )

    np.testing.assert_array_equal(
        calls[0]["priors"]["POR"],
        priors["POR"],
    )

    # The first forward evaluation becomes the only
    # intermediate checkpoint: M1, D1.
    (
        M_round_1,
        D_round_1,
        ids_round_1,
    ) = run.load_round_checkpoint(1)

    np.testing.assert_allclose(
        M_round_1,
        calls[0]["M"],
    )

    np.testing.assert_allclose(
        D_round_1,
        calls[0]["D"],
    )

    np.testing.assert_array_equal(
        ids_round_1,
        realization_ids,
    )

    # There is no round_002 checkpoint.
    assert not (
        run.state_path
        / "round_002.npz"
    ).exists()

    # The second forward evaluation is the final
    # posterior M2, D2.
    (
        M_post,
        D_post,
        ids_post,
    ) = run.load_post_checkpoint()

    np.testing.assert_allclose(
        M_post,
        calls[1]["M"],
    )

    np.testing.assert_allclose(
        D_post,
        calls[1]["D"],
    )

    np.testing.assert_array_equal(
        ids_post,
        realization_ids,
    )

    np.testing.assert_allclose(
        result.M,
        M_post,
    )

    np.testing.assert_allclose(
        result.D,
        D_post,
    )

    np.testing.assert_array_equal(
        result.realization_ids,
        realization_ids,
    )