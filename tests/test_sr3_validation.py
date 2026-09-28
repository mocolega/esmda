import h5py
import numpy as np
from datetime import datetime

from esmda4d.cmg.sr3 import SR3Reader

from esmda4d.cmg.sr3_validation import (
    SR3ProductionRequirement,
    SR3SpatialRequirement,
    SR3Validator,
)


def create_minimal_sr3(path):
    """
    Create the smallest SR3-like file needed
    for basic validation.
    """

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

        wells.create_dataset(
            "Data",
            data=np.array([
                [[100.0]],
                [[110.0]],
            ]),
        )

        # ---------------------------------------------
        # SpatialProperties
        # ---------------------------------------------

        spatial = file.create_group(
            "SpatialProperties"
        )

        for timestep, pressure, sw in [
            (
                "000000",
                [100.0, 110.0],
                [0.20, 0.30],
            ),
            (
                "000001",
                [120.0, 130.0],
                [0.40, 0.50],
            ),
        ]:
            state = spatial.create_group(
                timestep
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
                    [1],
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
                    [1, 2],
                    dtype=np.int32,
                ),
            )

            state.create_dataset(
                "PRESSURE",
                data=np.array(
                    pressure,
                    dtype=float,
                ),
            )

            state.create_dataset(
                "SW",
                data=np.array(
                    sw,
                    dtype=float,
                ),
            )


def test_sr3_validator_accepts_readable_file(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    validator = SR3Validator()

    result = validator.validate(
        model_path
    )

    assert result.valid
    assert result.sr3_path == sr3_path
    assert result.reason is None


def test_sr3_validator_rejects_missing_file(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    validator = SR3Validator()

    result = validator.validate(
        model_path
    )

    assert not result.valid

    assert result.sr3_path == (
        tmp_path / "model_0001.sr3"
    )

    assert result.reason == (
        "SR3 file not found."
    )


def test_sr3_validator_rejects_invalid_hdf5(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    sr3_path.write_text(
        "this is not an HDF5 file"
    )

    validator = SR3Validator()

    result = validator.validate(
        model_path
    )

    assert not result.valid

    assert result.sr3_path == sr3_path

    assert result.reason.startswith(
        "SR3 file could not be read:"
    )


def test_sr3_validator_rejects_missing_timetable(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    with h5py.File(
        sr3_path,
        "w",
    ) as file:
        file.create_group(
            "General"
        )

    validator = SR3Validator()

    result = validator.validate(
        model_path
    )

    assert not result.valid

    assert result.reason.startswith(
        "SR3 file could not be read:"
    )

def test_sr3_validator_accepts_required_production_data(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    reader = SR3Reader(
        sr3_path
    )

    dates = reader.dates(
        "WELLS"
    )

    requirements = [
        SR3ProductionRequirement(
            origin="WELLS",
            entity="WELL-1",
            variable="BHP",
            dates=dates,
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        production_requirements=requirements,
    )

    assert result.valid


def test_sr3_validator_rejects_missing_production_date(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    requirements = [
        SR3ProductionRequirement(
            origin="WELLS",
            entity="WELL-1",
            variable="BHP",
            dates=[
                datetime(2099, 1, 1),
            ],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        production_requirements=requirements,
    )

    assert not result.valid

    assert "Required production date" in (
        result.reason
    )


def test_sr3_validator_rejects_missing_production_variable(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    requirements = [
        SR3ProductionRequirement(
            origin="WELLS",
            entity="WELL-1",
            variable="NOT_A_VARIABLE",
            dates=[],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        production_requirements=requirements,
    )

    assert not result.valid

    assert (
        "Required production data "
        "could not be read"
        in result.reason
    )


def test_sr3_validator_rejects_nonfinite_production_value(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    # Replace the BHP value at timestep 1
    # with NaN.
    with h5py.File(
        sr3_path,
        "a",
    ) as file:
        data = file[
            "TimeSeries/WELLS/Data"
        ]

        data[1, 0, 0] = np.nan

    reader = SR3Reader(
        sr3_path
    )

    dates = reader.dates(
        "WELLS"
    )

    requirements = [
        SR3ProductionRequirement(
            origin="WELLS",
            entity="WELL-1",
            variable="BHP",
            dates=[
                dates[1],
            ],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        production_requirements=requirements,
    )

    assert not result.valid

    assert (
        "Required production value "
        "is not finite"
        in result.reason
    )


def test_sr3_validator_accepts_required_spatial_data(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    reader = SR3Reader(
        sr3_path
    )

    dates = reader.spatial_dates()

    requirements = [
        SR3SpatialRequirement(
            variable="PRESSURE",
            dates=dates,
        ),
        SR3SpatialRequirement(
            variable="SW",
            dates=dates,
        ),
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        spatial_requirements=requirements,
    )

    assert result.valid


def test_sr3_validator_rejects_missing_spatial_variable(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    reader = SR3Reader(
        sr3_path
    )

    dates = reader.spatial_dates()

    requirements = [
        SR3SpatialRequirement(
            variable="NOT_A_VARIABLE",
            dates=[
                dates[0],
            ],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        spatial_requirements=requirements,
    )

    assert not result.valid

    assert (
        "Required spatial data "
        "could not be read"
        in result.reason
    )

def test_sr3_validator_rejects_missing_spatial_date(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    requirements = [
        SR3SpatialRequirement(
            variable="PRESSURE",
            dates=[
                datetime(2099, 1, 1),
            ],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        spatial_requirements=requirements,
    )

    assert not result.valid

    assert (
        "Required spatial date"
        in result.reason
    )

    assert (
        "2099-01-01"
        in result.reason
    )

    assert (
        "not found"
        in result.reason
    )


def test_sr3_validator_rejects_nonfinite_active_spatial_value(
    tmp_path,
):
    model_path = (
        tmp_path / "model_0001.dat"
    )

    model_path.write_text(
        "fake CMG model"
    )

    sr3_path = (
        tmp_path / "model_0001.sr3"
    )

    create_minimal_sr3(
        sr3_path
    )

    with h5py.File(
        sr3_path,
        "a",
    ) as file:
        pressure = file[
            "SpatialProperties/"
            "000000/PRESSURE"
        ]

        pressure[1] = np.nan

    reader = SR3Reader(
        sr3_path
    )

    dates = reader.spatial_dates()

    requirements = [
        SR3SpatialRequirement(
            variable="PRESSURE",
            dates=[
                dates[0],
            ],
        )
    ]

    validator = SR3Validator()

    result = validator.validate(
        model_path,
        spatial_requirements=requirements,
    )

    assert not result.valid

    assert (
        "Required spatial data "
        "could not be read"
        in result.reason
    )

    assert (
        "non-finite values"
        in result.reason
    )