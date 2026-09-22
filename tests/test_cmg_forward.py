from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from esmda4d.cmg.forward import (
    CMGForwardModel,
    CMGRealizationFailure,
)


def create_sr3(
    path,
    bhp,
):
    with h5py.File(path, "w") as file:

        general = file.create_group(
            "General"
        )

        dtype = np.dtype([
            ("Index", np.int32),
            ("Date", np.float64),
        ])

        timetable = np.array([
            (0, 20200101.0),
            (1, 20200201.0),
        ], dtype=dtype)

        general.create_dataset(
            "MasterTimeTable",
            data=timetable,
        )

        time_series = file.create_group(
            "TimeSeries"
        )

        wells = time_series.create_group(
            "WELLS"
        )

        wells.create_dataset(
            "Timesteps",
            data=np.array(
                [0, 1],
                dtype=np.int32,
            ),
        )

        wells.create_dataset(
            "Variables",
            data=np.array([
                b"BHP",
            ]),
        )

        wells.create_dataset(
            "Origins",
            data=np.array([
                b"WELL-1",
            ]),
        )

        data = np.asarray(
            bhp,
            dtype=float,
        ).reshape(2, 1, 1)

        wells.create_dataset(
            "Data",
            data=data,
        )


class FakeWriter:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.last_realization_ids = None

    def write_ensemble(
        self,
        M,
        metadata,
        priors,
        realization_ids=None,
    ):
        if realization_ids is None:
            realization_ids = list(
                range(1, M.shape[1] + 1)
            )

        self.last_realization_ids = list(
            realization_ids
        )

        paths = []

        for j, realization_id in enumerate(
            realization_ids
        ):
            realization = (
                self.directory
                / (
                    "realization_"
                    f"{realization_id:04d}"
                )
            )

            realization.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_path = (
                realization
                / f"model_{realization_id:04d}.dat"
            )

            model_path.write_text(
                "FAKE CMG MODEL"
            )

            sr3_path = (
                model_path.with_suffix(
                    ".sr3"
                )
            )

            create_sr3(
                sr3_path,
                bhp=[
                    100.0 + j,
                    110.0 + j,
                ],
            )

            paths.append(
                model_path
            )

        return paths

class FakeRetryRunner:

    def __init__(self):
        self.calls = []
        self.attempts = {}

    def run_ensemble(
        self,
        model_paths,
    ):
        model_paths = list(model_paths)

        self.calls.append(
            [path.name for path in model_paths]
        )

        results = []

        for model_path in model_paths:

            name = model_path.name

            self.attempts[name] = (
                self.attempts.get(name, 0)
                + 1
            )

            # model_0002 fails only on its
            # first attempt.
            succeeded = not (
                name == "model_0002.dat"
                and self.attempts[name] == 1
            )

            results.append(
                SimpleNamespace(
                    succeeded=succeeded,
                )
            )

        return results

class FakePermanentFailureRunner:

    def __init__(self):
        self.calls = []

    def run_ensemble(
        self,
        model_paths,
    ):
        model_paths = list(model_paths)

        self.calls.append(
            [path.name for path in model_paths]
        )

        return [
            SimpleNamespace(
                succeeded=(
                    path.name
                    != "model_0002.dat"
                ),
            )
            for path in model_paths
        ]
    
class FakeRunner:

    def run_ensemble(
        self,
        model_paths,
    ):
        return [
            SimpleNamespace(
                succeeded=True
            )
            for _ in model_paths
        ]


def test_cmg_forward_model(
    tmp_path,
):
    writer = FakeWriter(
        tmp_path
    )

    runner = FakeRunner()

    model_metadata = [
        {
            "variable": "POR",
            "cell_id": 0,
        }
    ]

    production_metadata = [
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": np.datetime64(
                "2020-01-01"
            ),
        },
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": np.datetime64(
                "2020-02-01"
            ),
        },
    ]

    # One model parameter,
    # two realizations.

    M = np.array([
        [0.20, 0.25],
    ])

    priors = {
        "POR": np.array([
            [
                [
                    [0.20, 0.25]
                ]
            ]
        ])
    }

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=model_metadata,
        production_metadata=(
            production_metadata
        ),
        priors=priors,
    )

    # ForwardModel.__call__ should invoke run()

    D = forward(M)

    expected = np.array([
        [100.0, 101.0],
        [110.0, 111.0],
    ])

    np.testing.assert_allclose(
        D,
        expected,
    )

    assert D.shape == (2, 2)


def test_cmg_forward_model_preserves_realization_ids(
    tmp_path,
):
    writer = FakeWriter(
        tmp_path
    )

    runner = FakeRunner()

    model_metadata = [
        {
            "variable": "POR",
            "cell_id": 0,
        }
    ]

    production_metadata = [
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": np.datetime64(
                "2020-01-01"
            ),
        },
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": np.datetime64(
                "2020-02-01"
            ),
        },
    ]

    # One model parameter,
    # two realizations.

    M = np.array([
        [0.20, 0.25],
    ])

    priors = {
        "POR": np.array([
            [
                [
                    [0.20, 0.25]
                ]
            ]
        ])
    }

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=model_metadata,
        production_metadata=(
            production_metadata
        ),
        priors=priors,
        realization_ids=[
            2,
            7,
        ],
    )

    # ForwardModel.__call__ should invoke run()

    D = forward(M)

    assert writer.last_realization_ids == [
        2,
        7,
    ]

    assert (
        tmp_path
        / "realization_0002"
        / "model_0002.dat"
    ).is_file()

    assert (
        tmp_path
        / "realization_0007"
        / "model_0007.dat"
    ).is_file()

    assert not (
        tmp_path
        / "realization_0001"
    ).exists()

    np.testing.assert_allclose(
        D,
        np.array([
            [100.0, 101.0],
            [110.0, 111.0],
        ]),
    )

def test_cmg_forward_model_rejects_wrong_number_of_ids(
    tmp_path,
):
    writer = FakeWriter(
        tmp_path
    )

    runner = FakeRunner()

    M = np.array([
        [0.20, 0.25],
    ])

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=[],
        production_metadata=[],
        priors={},
        realization_ids=[1],
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "one ID",
    ):
        forward(M)

def test_cmg_forward_retries_only_failed_models(
    tmp_path,
):
    runner = FakeRetryRunner()

    forward = CMGForwardModel(
        writer=None,
        runner=runner,
        model_metadata=[],
        production_metadata=[],
        priors={},
        max_retries=1,
    )

    model_paths = []

    for realization_id in [
        1,
        2,
        4,
    ]:
        path = (
            tmp_path
            / f"model_{realization_id:04d}.dat"
        )

        path.write_text(
            "FAKE"
        )

        model_paths.append(
            path
        )

    results = forward._run_with_retries(
        model_paths
    )

    assert all(
        result.succeeded
        for result in results
    )

    assert runner.calls == [
        [
            "model_0001.dat",
            "model_0002.dat",
            "model_0004.dat",
        ],
        [
            "model_0002.dat",
        ],
    ]

    assert runner.attempts == {
        "model_0001.dat": 1,
        "model_0002.dat": 2,
        "model_0004.dat": 1,
    }

def test_cmg_forward_retry_keeps_permanent_failure(
    tmp_path,
):
    runner = FakePermanentFailureRunner()

    forward = CMGForwardModel(
        writer=None,
        runner=runner,
        model_metadata=[],
        production_metadata=[],
        priors={},
        max_retries=2,
    )

    model_paths = []

    for realization_id in [
        1,
        2,
        4,
    ]:
        path = (
            tmp_path
            / f"model_{realization_id:04d}.dat"
        )

        path.write_text(
            "FAKE"
        )

        model_paths.append(
            path
        )

    results = forward._run_with_retries(
        model_paths
    )

    assert results[0].succeeded
    assert not results[1].succeeded
    assert results[2].succeeded

    assert runner.calls == [
        [
            "model_0001.dat",
            "model_0002.dat",
            "model_0004.dat",
        ],
        [
            "model_0002.dat",
        ],
        [
            "model_0002.dat",
        ],
    ]

def test_cmg_forward_reports_failed_realization_ids(
    tmp_path,
):
    forward = CMGForwardModel(
        writer=None,
        runner=None,
        model_metadata=[],
        production_metadata=[],
        priors={},
    )

    results = [
        SimpleNamespace(succeeded=True),
        SimpleNamespace(succeeded=False),
        SimpleNamespace(succeeded=True),
        SimpleNamespace(succeeded=False),
    ]

    realization_ids = [
        1,
        3,
        7,
        12,
    ]

    try:
        forward._check_results(
            results=results,
            realization_ids=realization_ids,
        )

    except CMGRealizationFailure as error:
        assert error.failed_indices == [
            1,
            3,
        ]

        assert error.failed_ids == [
            3,
            12,
        ]

    else:
        raise AssertionError(
            "Expected CMGRealizationFailure."
        )

def test_cmg_forward_accepts_all_successful_results(
    tmp_path,
):
    forward = CMGForwardModel(
        writer=None,
        runner=None,
        model_metadata=[],
        production_metadata=[],
        priors={},
    )

    results = [
        SimpleNamespace(succeeded=True),
        SimpleNamespace(succeeded=True),
    ]

    forward._check_results(
        results=results,
        realization_ids=[
            4,
            9,
        ],
    )