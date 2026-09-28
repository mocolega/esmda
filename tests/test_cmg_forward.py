from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from esmda4d.processor import (
    DataProcessor,
    StateRequirement,
)

from esmda4d.run import evaluate_ensemble

from esmda4d.cmg.forward import (
    CMGForwardModel,
    CMGRealizationFailure,
)
import pytest

def create_sr3(
    path,
    bhp,
    pressure=None,
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

        if pressure is not None:
            spatial = file.create_group(
                "SpatialProperties"
            )

            state = spatial.create_group(
                "000000"
            )

            grid = state.create_group(
                "GRID"
            )

            grid.create_dataset(
                "IGNTID",
                data=np.array(
                    [2],
                    dtype=np.int32,
                ),
            )

            grid.create_dataset(
                "IGNTJD",
                data=np.array(
                    [2],
                    dtype=np.int32,
                ),
            )

            grid.create_dataset(
                "IGNTKD",
                data=np.array(
                    [1],
                    dtype=np.int32,
                ),
            )

            grid.create_dataset(
                "IPSTCS",
                data=np.array(
                    [1, 2, 3, 4],
                    dtype=np.int32,
                ),
            )

            state.create_dataset(
                "PRESSURE",
                data=np.asarray(
                    pressure,
                    dtype=float,
                ),
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
                pressure=[
                    1000.0 + 100.0 * j,
                    1100.0 + 100.0 * j,
                    1200.0 + 100.0 * j,
                    1300.0 + 100.0 * j,
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

def test_cmg_forward_model_with_processor(
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

    class FakeProcessor:
        def required_states(self):
            return []

        def run(self, states):
            return np.array([
                10.0,
                20.0,
            ])

    processor = FakeProcessor()

    class FakeStateReader:
        def read(
            self,
            model_path,
            requirements,
        ):
            return {}

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=model_metadata,
        production_metadata=(
            production_metadata
        ),
        priors=priors,
        processor=processor,
    )

    forward.state_reader = (
        FakeStateReader()
    )

    # ForwardModel.__call__ should invoke run()

    D = forward(M)

    expected = np.array([
        [100.0, 101.0],
        [110.0, 111.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ])

    np.testing.assert_allclose(
        D,
        expected,
    )

    assert D.shape == (4, 2)

def test_cmg_forward_model_reads_real_sr3_states(
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

    date = np.datetime64(
        "2020-01-01"
    )

    class PressureProcessor(
        DataProcessor
    ):
        def required_states(self):
            return [
                StateRequirement(
                    variable="PRESSURE",
                    dates=[date],
                )
            ]

        def run(self, states):
            pressure = states[
                "PRESSURE"
            ][date]

            return np.array([
                np.mean(pressure)
            ])

    processor = PressureProcessor()

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=model_metadata,
        production_metadata=(
            production_metadata
        ),
        priors=priors,
        processor=processor,
    )

    D = forward(M)

    expected = np.array([
        [100.0, 101.0],
        [110.0, 111.0],
        [1150.0, 1250.0],
    ])

    np.testing.assert_allclose(
        D,
        expected,
    )

    assert D.shape == (3, 2)


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

    forward._validate_sr3_results = (
        lambda model_paths, results: [
            result.succeeded
            for result in results
        ]
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

def test_cmg_forward_retries_invalid_sr3(
        tmp_path,
    ):
        class SuccessfulRunner:

            def __init__(self):
                self.calls = []

            def run_ensemble(
                self,
                model_paths,
            ):
                model_paths = list(
                    model_paths
                )

                self.calls.append([
                    path.name
                    for path in model_paths
                ])

                return [
                    SimpleNamespace(
                        succeeded=True
                    )
                    for _ in model_paths
                ]

        runner = SuccessfulRunner()

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

        validation_calls = []

        def fake_validate_sr3_results(
            paths,
            results,
        ):
            validation_calls.append([
                path.name
                for path in paths
            ])

            if len(validation_calls) == 1:
                return [
                    True,
                    False,
                    True,
                ]

            return [
                True
                for _ in paths
            ]

        forward._validate_sr3_results = (
            fake_validate_sr3_results
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

        assert validation_calls == [
            [
                "model_0001.dat",
                "model_0002.dat",
                "model_0004.dat",
            ],
        ]


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

    forward._validate_sr3_results = (
        lambda model_paths, results: [
            result.succeeded
            for result in results
        ]
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
        forward._validate_sr3_results = (
            lambda model_paths, results: [
                True,
                False,
                True,
                False,
            ]
        )
        model_paths = [
            tmp_path / "model_0001.dat",
            tmp_path / "model_0003.dat",
            tmp_path / "model_0007.dat",
            tmp_path / "model_0012.dat",
        ]
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
                model_paths=model_paths,
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
        forward._validate_sr3_results = (
            lambda model_paths, results: [
                True,
                True,
            ]
        )
        model_paths = [
            tmp_path / "model_0004.dat",
            tmp_path / "model_0009.dat",
        ]
        results = [
            SimpleNamespace(succeeded=True),
            SimpleNamespace(succeeded=True),
        ]

        forward._check_results(
            model_paths=model_paths,
            results=results,
            realization_ids=[
                4,
                9,
            ],
        )

def test_cmg_forward_updates_ensemble_context():
    forward = CMGForwardModel(
        writer=None,
        runner=None,
        model_metadata=[],
        production_metadata=[],
        priors={},
    )

    priors = {
        "POR": np.zeros(
            (2, 2, 1, 3)
        ),
        "PERM": np.zeros(
            (2, 2, 1, 3)
        ),
    }

    forward.set_ensemble_context(
        priors=priors,
        realization_ids=[
            1,
            7,
            12,
        ],
    )

    assert forward.priors is priors

    assert forward.realization_ids == [
        1,
        7,
        12,
    ]

def test_cmg_forward_context_rejects_wrong_prior_size():
    forward = CMGForwardModel(
        writer=None,
        runner=None,
        model_metadata=[],
        production_metadata=[],
        priors={},
    )

    priors = {
        "POR": np.zeros(
            (2, 2, 1, 4)
        ),
    }

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        forward.set_ensemble_context(
            priors=priors,
            realization_ids=[
                1,
                7,
                12,
            ],
        )

def test_cmg_forward_context_rejects_duplicate_ids():
    forward = CMGForwardModel(
        writer=None,
        runner=None,
        model_metadata=[],
        production_metadata=[],
        priors={},
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
        forward.set_ensemble_context(
            priors=priors,
            realization_ids=[
                1,
                7,
                7,
            ],
        )

class FakeExclusionRunner:
    """
    Realization 3 always fails.
    All other realizations succeed.
    """

    def __init__(self):
        self.calls = []

    def run_ensemble(self, model_paths):
        model_paths = list(model_paths)

        self.calls.append([
            path.name
            for path in model_paths
        ])

        results = []

        for path in model_paths:
            failed = (
                "0003" in path.name
            )

            results.append(
                SimpleNamespace(
                    succeeded=not failed
                )
            )

        return results

def test_evaluate_ensemble_with_cmg_failure(
    tmp_path,
):
    writer = FakeWriter(
        tmp_path
    )

    runner = FakeExclusionRunner()

    priors = {
        "POR": np.zeros(
            (1, 1, 1, 4)
        ),
    }

    forward = CMGForwardModel(
        writer=writer,
        runner=runner,
        model_metadata=[],
        production_metadata=[],
        priors=priors,
        realization_ids=[
            1,
            3,
            7,
            12,
        ],
        max_retries=0,
    )

    M = np.array([
        [10.0, 20.0, 30.0, 40.0],
    ])

    with pytest.raises(
        ValueError,
        match="Production metadata cannot be empty",
    ):
        evaluate_ensemble(
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
                CMGRealizationFailure
            ),
        )

    assert writer.last_realization_ids == [
        1,
        7,
        12,
    ]

    assert forward.realization_ids == [
        1,
        7,
        12,
    ]

    assert forward.priors["POR"].shape[-1] == 3

def test_builds_production_requirements():
    forward_model = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProductionReader:
        def _get_origin(
            self,
            entity_type,
        ):
            return {
                "well": "WELLS",
                "sector": "SECTORS",
            }[entity_type]

    forward_model.production_reader = (
        FakeProductionReader()
    )

    

    forward_model.production_metadata = [
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": "2020-01-01",
        },
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": "2020-02-01",
        },
        {
            "entity": "WELL-2",
            "entity_type": "well",
            "variable": "WOPR",
            "time": "2020-01-01",
        },
    ]

    requirements = (
        forward_model
        ._production_requirements()
    )

    assert len(requirements) == 2

    first = requirements[0]

    assert first.origin == "WELLS"
    assert first.entity == "WELL-1"
    assert first.variable == "BHP"

    assert first.dates == [
        "2020-01-01",
        "2020-02-01",
    ]

def test_validate_sr3_results_uses_required_production_data(
    tmp_path,
):
    forward_model = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProductionReader:
        def _get_origin(
            self,
            entity_type,
        ):
            return {
                "well": "WELLS",
                "sector": "SECTORS",
            }[entity_type]

    forward_model.production_reader = (
        FakeProductionReader()
    )



    class FakeSR3Validation:
        def __init__(self, valid):
            self.valid = valid

    class FakeSR3Validator:
        def __init__(self):
            self.calls = []

        def validate(
            self,
            model_path,
            production_requirements=None,
            spatial_requirements=None,
        ):
            self.calls.append(
                (
                    model_path,
                    production_requirements,
                )
            )

            return FakeSR3Validation(
                valid=True
            )

    class FakeResult:
        succeeded = True

    forward_model.production_reader = (
        FakeProductionReader()
    )

    forward_model.production_metadata = [
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": "2020-01-01",
        },
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": "2020-02-01",
        },
    ]

    forward_model.sr3_validator = (
        FakeSR3Validator()
    )

    model_paths = [
        tmp_path / "model_0001.dat",
        tmp_path / "model_0002.dat",
    ]

    results = [
        FakeResult(),
        FakeResult(),
    ]

    valid = (
        forward_model
        ._validate_sr3_results(
            model_paths,
            results,
        )
    )

    assert valid == [
        True,
        True,
    ]

    assert len(
        forward_model
        .sr3_validator.calls
    ) == 2

    _, requirements = (
        forward_model
        .sr3_validator.calls[0]
    )

    assert len(requirements) == 1

    requirement = requirements[0]

    assert requirement.origin == "WELLS"
    assert requirement.entity == "WELL-1"
    assert requirement.variable == "BHP"

    assert requirement.dates == [
        "2020-01-01",
        "2020-02-01",
    ]

def test_validate_sr3_results_rejects_invalid_sr3(
    tmp_path,
):
    forward_model = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProductionReader:
        def _get_origin(
            self,
            entity_type,
        ):
            return {
                "well": "WELLS",
                "sector": "SECTORS",
            }[entity_type]

    forward_model.production_reader = (
        FakeProductionReader()
    )


    class FakeSR3Validation:
        def __init__(self, valid):
            self.valid = valid

    class FakeSR3Validator:
        def __init__(self):
            self.call_count = 0

        def validate(
            self,
            model_path,
            production_requirements=None,
            spatial_requirements=None,
        ):
            self.call_count += 1

            return FakeSR3Validation(
                valid=(
                    self.call_count == 1
                )
            )

    class FakeResult:
        succeeded = True

    forward_model.production_reader = (
        FakeProductionReader()
    )

    forward_model.production_metadata = [
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "BHP",
            "time": "2020-01-01",
        },
    ]

    forward_model.sr3_validator = (
        FakeSR3Validator()
    )

    model_paths = [
        tmp_path / "model_0001.dat",
        tmp_path / "model_0002.dat",
    ]

    results = [
        FakeResult(),
        FakeResult(),
    ]

    valid = (
        forward_model
        ._validate_sr3_results(
            model_paths,
            results,
        )
    )

    assert valid == [
        True,
        False,
    ]

def test_validate_sr3_results_skips_failed_cmg_run(
    tmp_path,
):
    forward_model = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProductionReader:
        def _get_origin(
            self,
            entity_type,
        ):
            return {
                "well": "WELLS",
                "sector": "SECTORS",
            }[entity_type]

    forward_model.production_reader = (
        FakeProductionReader()
    )



    class FakeSR3Validation:
        valid = True

    class FakeSR3Validator:
        def __init__(self):
            self.call_count = 0

        def validate(
            self,
            model_path,
            production_requirements=None,
            spatial_requirements=None,
        ):
            self.call_count += 1
            return FakeSR3Validation()

    class FakeResult:
        def __init__(self, succeeded):
            self.succeeded = succeeded

    forward_model.production_reader = (
        FakeProductionReader()
    )

    forward_model.production_metadata = []

    forward_model.sr3_validator = (
        FakeSR3Validator()
    )

    model_paths = [
        tmp_path / "model_0001.dat",
        tmp_path / "model_0002.dat",
    ]

    results = [
        FakeResult(False),
        FakeResult(True),
    ]

    valid = (
        forward_model
        ._validate_sr3_results(
            model_paths,
            results,
        )
    )

    assert valid == [
        False,
        True,
    ]

    assert (
        forward_model
        .sr3_validator.call_count
        == 1
    )

# def test_debug_fake_writer_sr3(
#     tmp_path,
# ):
#     writer = FakeWriter(
#         tmp_path
#     )

#     M = np.array([
#         [0.20, 0.25],
#     ])

#     model_paths = writer.write_ensemble(
#         M=M,
#         metadata=[],
#         priors={},
#         realization_ids=[
#             1,
#             2,
#         ],
#     )

#     forward = CMGForwardModel(
#         writer=writer,
#         runner=FakeRunner(),
#         model_metadata=[],
#         production_metadata=[
#             {
#                 "entity": "WELL-1",
#                 "entity_type": "well",
#                 "variable": "BHP",
#                 "time": np.datetime64(
#                     "2020-01-01"
#                 ),
#             },
#             {
#                 "entity": "WELL-1",
#                 "entity_type": "well",
#                 "variable": "BHP",
#                 "time": np.datetime64(
#                     "2020-02-01"
#                 ),
#             },
#         ],
#         priors={},
#     )

#     from esmda4d.cmg.sr3 import SR3Reader

#     reader = SR3Reader(
#         model_paths[0].with_suffix(
#             ".sr3"
#         )
#     )

#     dates, values = (
#         reader.read_time_series(
#             origin="WELLS",
#             variable="BHP",
#             entity="WELL-1",
#         )
#     )

#     print("dates =", dates)
#     print("dates type =", type(dates))

#     for date in dates:
#         print(
#             "date:",
#             repr(date),
#             "type:",
#             type(date),
#         )

#     print(
#         "required:",
#         repr(
#             np.datetime64(
#                 "2020-01-01"
#             )
#         ),
#         type(
#             np.datetime64(
#                 "2020-01-01"
#             )
#         ),
#     )
#     requirements = (
#         forward._production_requirements()
#     )

#     for model_path in model_paths:
#         validation = (
#             forward.sr3_validator.validate(
#                 model_path,
#                 production_requirements=(
#                     requirements
#                 ),
#             )
#         )

#         print(
#             model_path.name,
#             validation.valid,
#             validation.reason,
#         )

#         assert validation.valid

class FakeProcessor(DataProcessor):

    def required_states(self):
        return [
            StateRequirement(
                variable="PRESSURE",
                dates=[
                    np.datetime64(
                        "2020-01-01"
                    ),
                    np.datetime64(
                        "2022-01-01"
                    ),
                ],
            ),
            StateRequirement(
                variable="SW",
                dates=[
                    np.datetime64(
                        "2022-01-01"
                    ),
                ],
            ),
        ]

    def run(self, states):
        raise NotImplementedError

def test_cmg_forward_builds_spatial_requirements():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    forward.processor = FakeProcessor()

    requirements = (
        forward._spatial_requirements()
    )

    assert len(requirements) == 2

    assert requirements[0].variable == (
        "PRESSURE"
    )

    assert requirements[0].dates == [
        np.datetime64("2020-01-01"),
        np.datetime64("2022-01-01"),
    ]

    assert requirements[1].variable == "SW"

    assert requirements[1].dates == [
        np.datetime64("2022-01-01"),
    ]

def test_cmg_forward_has_no_spatial_requirements_without_processor():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    forward.processor = None

    assert (
        forward._spatial_requirements()
        == []
    )

def test_cmg_forward_processes_derived_data():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    date = np.datetime64(
        "2020-01-01"
    )

    pressure = np.arange(
        8,
        dtype=float,
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    class FakeProcessor:
        def required_states(self):
            return [
                StateRequirement(
                    variable="PRESSURE",
                    dates=[date],
                ),
            ]

        def run(self, states):
            return states[
                "PRESSURE"
            ][date].ravel(
                order="F"
            )

    class FakeStateReader:
        def read(
            self,
            model_path,
            requirements,
        ):
            return {
                "PRESSURE": {
                    date: pressure,
                },
            }

    forward.processor = FakeProcessor()
    forward.state_reader = (
        FakeStateReader()
    )

    result = (
        forward._process_derived_data(
            model_path="model_0001.dat",
        )
    )

    np.testing.assert_array_equal(
        result,
        np.arange(
            8,
            dtype=float,
        ),
    )

def test_cmg_forward_has_no_derived_data_without_processor():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    forward.processor = None

    result = (
        forward._process_derived_data(
            model_path="model_0001.dat",
        )
    )

    assert result.shape == (0,)

def test_cmg_forward_rejects_non_1d_processor_output():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProcessor:
        def required_states(self):
            return []

        def run(self, states):
            return np.ones(
                (2, 2)
            )

    class FakeStateReader:
        def read(
            self,
            model_path,
            requirements,
        ):
            return {}

    forward.processor = FakeProcessor()
    forward.state_reader = (
        FakeStateReader()
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        forward._process_derived_data(
            model_path="model_0001.dat",
        )

def test_cmg_forward_rejects_nonfinite_processor_output():
    forward = CMGForwardModel.__new__(
        CMGForwardModel
    )

    class FakeProcessor:
        def required_states(self):
            return []

        def run(self, states):
            return np.array([
                1.0,
                np.nan,
            ])

    class FakeStateReader:
        def read(
            self,
            model_path,
            requirements,
        ):
            return {}

    forward.processor = FakeProcessor()
    forward.state_reader = (
        FakeStateReader()
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        forward._process_derived_data(
            model_path="model_0001.dat",
        )