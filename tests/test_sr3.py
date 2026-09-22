import datetime as dt

import h5py
import numpy as np
import pytest

from esmda4d.cmg.sr3 import SR3Reader


@pytest.fixture
def sr3_path(tmp_path):
    """
    Create a minimal synthetic SR3-like HDF5 file.

    Structure:

    /General/MasterTimeTable

    /TimeSeries/WELLS/
        Timesteps
        Variables
        Origins
        Data
    """

    path = tmp_path / "synthetic.sr3"

    with h5py.File(path, "w") as file:

        # ---------------------------------------------
        # Master timetable
        # ---------------------------------------------

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
            (2, 20200301.0),
        ], dtype=dtype)

        general.create_dataset(
            "MasterTimeTable",
            data=timetable,
        )

        # ---------------------------------------------
        # TimeSeries/WELLS
        # ---------------------------------------------

        time_series = file.create_group(
            "TimeSeries"
        )

        wells = time_series.create_group(
            "WELLS"
        )

        wells.create_dataset(
            "Timesteps",
            data=np.array(
                [0, 1, 2],
                dtype=np.int32,
            ),
        )

        wells.create_dataset(
            "Variables",
            data=np.array([
                b"BHP",
                b"OILRATSC",
                b"WATRATSC",
            ]),
        )

        wells.create_dataset(
            "Origins",
            data=np.array([
                b"WELL-1",
                b"WELL-2",
            ]),
        )

        # Data shape:
        #
        # (n_times, n_variables, n_entities)
        #
        # time 0:
        # BHP      -> 100, 200
        # OILRATSC -> 10, 20
        # WATRATSC -> 1, 2
        #
        # etc.

        data = np.array([
            [
                [100.0, 200.0],
                [10.0, 20.0],
                [1.0, 2.0],
            ],
            [
                [110.0, 210.0],
                [11.0, 21.0],
                [1.1, 2.1],
            ],
            [
                [120.0, 220.0],
                [12.0, 22.0],
                [1.2, 2.2],
            ],
        ])

        wells.create_dataset(
            "Data",
            data=data,
        )

    return path


@pytest.fixture
def reader(sr3_path):
    return SR3Reader(sr3_path)


def test_sr3_file_exists(reader):
    assert reader.path.is_file()


def test_time_series_contains_wells(reader):
    origins = reader.time_series_origins()

    assert "WELLS" in origins


def test_well_variables(reader):
    variables = reader.variables(
        "WELLS"
    )

    assert variables == [
        "BHP",
        "OILRATSC",
        "WATRATSC",
    ]


def test_well_entities(reader):
    entities = reader.entities(
        "WELLS"
    )

    assert entities == [
        "WELL-1",
        "WELL-2",
    ]


def test_master_timetable(reader):
    timetable = reader.master_timetable()

    assert timetable[0] == dt.datetime(
        2020,
        1,
        1,
    )

    assert timetable[1] == dt.datetime(
        2020,
        2,
        1,
    )

    assert timetable[2] == dt.datetime(
        2020,
        3,
        1,
    )


def test_well_dates(reader):
    dates = reader.dates(
        "WELLS"
    )

    assert dates == [
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 2, 1),
        dt.datetime(2020, 3, 1),
    ]


def test_read_bhp_time_series(reader):
    dates, values = (
        reader.read_time_series(
            origin="WELLS",
            variable="BHP",
            entity="WELL-1",
        )
    )

    assert dates == [
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 2, 1),
        dt.datetime(2020, 3, 1),
    ]

    np.testing.assert_allclose(
        values,
        [100.0, 110.0, 120.0],
    )


def test_read_second_well(reader):
    _, values = reader.read_time_series(
        origin="WELLS",
        variable="OILRATSC",
        entity="WELL-2",
    )

    np.testing.assert_allclose(
        values,
        [20.0, 21.0, 22.0],
    )


def test_unknown_variable_raises_error(reader):
    with pytest.raises(
        ValueError,
        match="Variable",
    ):
        reader.read_time_series(
            origin="WELLS",
            variable="NOT_A_VARIABLE",
            entity="WELL-1",
        )


def test_unknown_entity_raises_error(reader):
    with pytest.raises(
        ValueError,
        match="Entity",
    ):
        reader.read_time_series(
            origin="WELLS",
            variable="BHP",
            entity="NOT_A_WELL",
        )


def test_unknown_origin_raises_error(reader):
    with pytest.raises(
        ValueError,
        match="origin",
    ):
        reader.variables(
            "NOT_AN_ORIGIN"
        )