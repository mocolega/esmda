import datetime as dt
from datetime import datetime
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

    /SpatialProperties/
        000000/
            GRID/
            PRESSURE
            SW

        000002/
            GRID/
            PRESSURE
            SW

    Time-series data are available at all three
    master timesteps.

    Spatial data are available only at timesteps
    0 and 2.
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

                # ---------------------------------------------
        # SpatialProperties
        # ---------------------------------------------
        #
        # Grid dimensions:
        #
        # NI = 2
        # NJ = 2
        # NK = 2
        #
        # CMG cell numbering:
        #
        # cell = i + NI*j + NI*NJ*k + 1
        #
        # Therefore:
        #
        # cell 1 -> (0, 0, 0)
        # cell 2 -> (1, 0, 0)
        # cell 3 -> (0, 1, 0)
        # cell 4 -> (1, 1, 0)
        # cell 5 -> (0, 0, 1)
        # cell 6 -> (1, 0, 1)
        # cell 7 -> (0, 1, 1)
        # cell 8 -> (1, 1, 1)
        #
        # Cell 4 is inactive.
        # ---------------------------------------------

        spatial = file.create_group(
            "SpatialProperties"
        )

        active_cells = np.array(
            [1, 2, 3, 5, 6, 7, 8],
            dtype=np.int32,
        )

        # ---------------------------------------------
        # Spatial timestep 0
        # ---------------------------------------------

        state_0 = spatial.create_group(
            "000000"
        )

        grid_0 = state_0.create_group(
            "GRID"
        )

        grid_0.create_dataset(
            "IGNTID",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_0.create_dataset(
            "IGNTJD",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_0.create_dataset(
            "IGNTKD",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_0.create_dataset(
            "IPSTCS",
            data=active_cells,
        )

        state_0.create_dataset(
            "PRESSURE",
            data=np.array([
                101.0,
                102.0,
                103.0,
                105.0,
                106.0,
                107.0,
                108.0,
            ]),
        )

        state_0.create_dataset(
            "SW",
            data=np.array([
                0.1,
                0.2,
                0.3,
                0.5,
                0.6,
                0.7,
                0.8,
            ]),
        )

        # ---------------------------------------------
        # Spatial timestep 2
        # ---------------------------------------------

        state_2 = spatial.create_group(
            "000002"
        )

        grid_2 = state_2.create_group(
            "GRID"
        )

        grid_2.create_dataset(
            "IGNTID",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_2.create_dataset(
            "IGNTJD",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_2.create_dataset(
            "IGNTKD",
            data=np.array(
                [2],
                dtype=np.int32,
            ),
        )

        grid_2.create_dataset(
            "IPSTCS",
            data=active_cells,
        )

        state_2.create_dataset(
            "PRESSURE",
            data=np.array([
                201.0,
                202.0,
                203.0,
                205.0,
                206.0,
                207.0,
                208.0,
            ]),
        )

        state_2.create_dataset(
            "SW",
            data=np.array([
                0.11,
                0.21,
                0.31,
                0.51,
                0.61,
                0.71,
                0.81,
            ]),
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

def test_sr3_spatial_dates(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    dates = reader.spatial_dates()

    assert dates == [
        timetable[0],
        timetable[2],
    ]


def test_sr3_spatial_variables(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    variables = reader.spatial_variables(
        timetable[0]
    )

    assert set(variables) == {
        "PRESSURE",
        "SW",
    }


def test_sr3_spatial_variables_rejects_date_without_grid_data(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    with pytest.raises(
        ValueError,
        match="No spatial data stored",
    ):
        reader.spatial_variables(
            timetable[1]
        )


def test_sr3_spatial_variables_rejects_unknown_date(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    with pytest.raises(
        ValueError,
        match="not found in MasterTimeTable",
    ):
        reader.spatial_variables(
            datetime(2099, 1, 1)
        )

def test_read_spatial_property(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    pressure = reader.read_spatial_property(
        variable="PRESSURE",
        date=timetable[0],
    )

    assert pressure.shape == (
        2,
        2,
        2,
    )

    expected = np.array([
        [
            [101.0, 105.0],
            [103.0, 107.0],
        ],
        [
            [102.0, 106.0],
            [np.nan, 108.0],
        ],
    ])

    np.testing.assert_allclose(
        pressure,
        expected,
        equal_nan=True,
    )

def test_read_spatial_property_rejects_unknown_variable(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    with pytest.raises(
        ValueError,
        match="Spatial variable",
    ):
        reader.read_spatial_property(
            variable="NOT_A_VARIABLE",
            date=timetable[0],
        )


def test_read_spatial_property_rejects_date_without_spatial_data(
    sr3_path,
):
    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    with pytest.raises(
        ValueError,
        match="No spatial data stored",
    ):
        reader.read_spatial_property(
            variable="PRESSURE",
            date=timetable[1],
        )

def test_read_spatial_property_rejects_inconsistent_sizes(
    sr3_path,
):
    """
    Spatial property values must correspond
    one-to-one with IPSTCS entries.
    """

    with h5py.File(
        sr3_path,
        "a",
    ) as file:
        state = file[
            "SpatialProperties/000000"
        ]

        del state["PRESSURE"]

        state.create_dataset(
            "PRESSURE",
            data=np.array([
                101.0,
                102.0,
                103.0,
            ]),
        )

    reader = SR3Reader(
        sr3_path
    )

    timetable = reader.master_timetable()

    with pytest.raises(
        ValueError,
        match="inconsistent sizes",
    ):
        reader.read_spatial_property(
            variable="PRESSURE",
            date=timetable[0],
        )


def test_read_spatial_property_rejects_nonfinite_active_value(
    sr3_path,
):
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

    timetable = reader.master_timetable()

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        reader.read_spatial_property(
            variable="PRESSURE",
            date=timetable[0],
        )