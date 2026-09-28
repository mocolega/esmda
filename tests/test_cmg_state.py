import datetime as dt

import numpy as np

from esmda4d.processor import StateRequirement
from esmda4d.cmg.state import CMGStateReader


class FakeSR3Reader:

    def __init__(self):
        self.calls = []

    def spatial_dates(self):
        return [
            dt.datetime(2020, 1, 1),
            dt.datetime(2022, 1, 1),
        ]

    def read_spatial_property(
        self,
        variable,
        date,
    ):
        self.calls.append(
            (variable, date)
        )

        value = {
            ("PRESSURE", 2020): 100.0,
            ("PRESSURE", 2022): 120.0,
            ("SW", 2020): 0.20,
            ("SW", 2022): 0.30,
        }[
            variable,
            date.year,
        ]

        return np.full(
            (2, 3, 2),
            value,
        )


def test_cmg_state_reader_reads_required_states(
    monkeypatch,
    tmp_path,
):
    fake_reader = FakeSR3Reader()

    monkeypatch.setattr(
        "esmda4d.cmg.state.SR3Reader",
        lambda path: fake_reader,
    )

    model_path = (
        tmp_path / "model_0001.dat"
    )

    requirements = [
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
                    "2020-01-01"
                ),
                np.datetime64(
                    "2022-01-01"
                ),
            ],
        ),
    ]

    state_reader = CMGStateReader()

    states = state_reader.read(
        model_path=model_path,
        requirements=requirements,
    )

    assert set(states) == {
        "PRESSURE",
        "SW",
    }

    assert states[
        "PRESSURE"
    ][
        np.datetime64("2020-01-01")
    ].shape == (
        2,
        3,
        2,
    )

    np.testing.assert_allclose(
        states[
            "PRESSURE"
        ][
            np.datetime64("2020-01-01")
        ],
        100.0,
    )

    np.testing.assert_allclose(
        states[
            "PRESSURE"
        ][
            np.datetime64("2022-01-01")
        ],
        120.0,
    )

    np.testing.assert_allclose(
        states[
            "SW"
        ][
            np.datetime64("2020-01-01")
        ],
        0.20,
    )

    np.testing.assert_allclose(
        states[
            "SW"
        ][
            np.datetime64("2022-01-01")
        ],
        0.30,
    )

def test_cmg_state_reader_rejects_missing_date(
    monkeypatch,
    tmp_path,
):
    fake_reader = FakeSR3Reader()

    monkeypatch.setattr(
        "esmda4d.cmg.state.SR3Reader",
        lambda path: fake_reader,
    )

    requirements = [
        StateRequirement(
            variable="PRESSURE",
            dates=[
                np.datetime64(
                    "2025-01-01"
                ),
            ],
        ),
    ]

    state_reader = CMGStateReader()

    with np.testing.assert_raises_regex(
        ValueError,
        "Required spatial date",
    ):
        state_reader.read(
            model_path=(
                tmp_path
                / "model_0001.dat"
            ),
            requirements=requirements,
        )