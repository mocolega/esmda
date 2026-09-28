import pytest
import numpy as np

from esmda4d.processor import (
    DataProcessor,
    StateRequirement,
)


def test_state_requirement():
    requirement = StateRequirement(
        variable="PRESSURE",
        dates=[
            "2020-01-01",
            "2022-01-01",
        ],
    )

    assert requirement.variable == "PRESSURE"

    assert requirement.dates == [
        "2020-01-01",
        "2022-01-01",
    ]


def test_data_processor_requires_state_implementation():
    processor = DataProcessor()

    with pytest.raises(
        NotImplementedError
    ):
        processor.required_states()


def test_data_processor_requires_run_implementation():
    processor = DataProcessor()

    with pytest.raises(
        NotImplementedError
    ):
        processor.run(
            states={}
        )

class ExampleProcessor(DataProcessor):

    def required_states(self):
        return [
            StateRequirement(
                variable="PRESSURE",
                dates=[
                    np.datetime64(
                        "2020-01-01"
                    ),
                ],
            ),
        ]

    def run(self, states):
        pressure = states[
            "PRESSURE"
        ][
            np.datetime64("2020-01-01")
        ]

        return pressure.ravel(
            order="F"
        )

def test_data_processor_can_return_data_vector():
    processor = ExampleProcessor()

    pressure = np.arange(
        8,
        dtype=float,
    ).reshape(
        (2, 2, 2),
        order="F",
    )

    states = {
        "PRESSURE": {
            np.datetime64(
                "2020-01-01"
            ): pressure,
        },
    }

    result = processor.run(
        states
    )

    assert isinstance(
        result,
        np.ndarray,
    )

    assert result.ndim == 1

    np.testing.assert_array_equal(
        result,
        np.arange(
            8,
            dtype=float,
        ),
    )