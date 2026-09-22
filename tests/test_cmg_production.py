import h5py
import numpy as np
import pytest

from esmda4d.cmg.production import (
    CMGProductionReader,
)


def create_sr3(
    path,
    bhp_well_1,
    oil_well_1,
):
    """
    Create a minimal SR3-like file containing
    two wells and two variables.
    """

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
            ]),
        )

        wells.create_dataset(
            "Origins",
            data=np.array([
                b"WELL-1",
                b"WELL-2",
            ]),
        )

        data = np.zeros(
            (3, 2, 2),
            dtype=float,
        )

        # WELL-1
        data[:, 0, 0] = bhp_well_1
        data[:, 1, 0] = oil_well_1

        # WELL-2
        data[:, 0, 1] = [
            200.0,
            210.0,
            220.0,
        ]

        data[:, 1, 1] = [
            20.0,
            21.0,
            22.0,
        ]

        wells.create_dataset(
            "Data",
            data=data,
        )


@pytest.fixture
def sr3_ensemble(tmp_path):

    path1 = tmp_path / "model_0001.sr3"
    path2 = tmp_path / "model_0002.sr3"

    create_sr3(
        path1,
        bhp_well_1=[
            100.0,
            110.0,
            120.0,
        ],
        oil_well_1=[
            10.0,
            11.0,
            12.0,
        ],
    )

    create_sr3(
        path2,
        bhp_well_1=[
            101.0,
            111.0,
            121.0,
        ],
        oil_well_1=[
            15.0,
            16.0,
            17.0,
        ],
    )

    return [
        path1,
        path2,
    ]


@pytest.fixture
def metadata():
    """
    Same structure produced by
    build_production_observations().
    """

    return [
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
        {
            "entity": "WELL-1",
            "entity_type": "well",
            "variable": "OILRATSC",
            "time": np.datetime64(
                "2020-01-01"
            ),
        },
    ]


def test_origin_mapping():
    reader = CMGProductionReader()

    assert (
        reader._get_origin("well")
        == "WELLS"
    )

    assert (
        reader._get_origin("sector")
        == "SECTORS"
    )


def test_unknown_entity_type():
    reader = CMGProductionReader()

    with pytest.raises(
        ValueError,
        match="origin mapping",
    ):
        reader._get_origin(
            "unknown"
        )


def test_read_ensemble(
    sr3_ensemble,
    metadata,
):
    reader = CMGProductionReader()

    simulated = reader.read_ensemble(
        sr3_paths=sr3_ensemble,
        metadata=metadata,
    )

    # Two unique datasets:
    #
    # WELL-1 / BHP
    # WELL-1 / OILRATSC

    assert len(simulated) == 2

    bhp = simulated[0]
    oil = simulated[1]

    assert bhp.entity == "WELL-1"
    assert bhp.entity_type == "well"
    assert bhp.variable == "BHP"

    np.testing.assert_allclose(
        bhp.values,
        [
            [100.0, 101.0],
            [110.0, 111.0],
            [120.0, 121.0],
        ],
    )

    np.testing.assert_allclose(
        oil.values,
        [
            [10.0, 15.0],
            [11.0, 16.0],
            [12.0, 17.0],
        ],
    )


def test_read_ensemble_shape(
    sr3_ensemble,
    metadata,
):
    reader = CMGProductionReader()

    simulated = reader.read_ensemble(
        sr3_paths=sr3_ensemble,
        metadata=metadata,
    )

    assert simulated[0].values.shape == (
        3,
        2,
    )


def test_requires_two_realizations(
    sr3_ensemble,
    metadata,
):
    reader = CMGProductionReader()

    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        reader.read_ensemble(
            sr3_paths=[
                sr3_ensemble[0]
            ],
            metadata=metadata,
        )


def test_requires_metadata(
    sr3_ensemble,
):
    reader = CMGProductionReader()

    with pytest.raises(
        ValueError,
        match="metadata",
    ):
        reader.read_ensemble(
            sr3_paths=sr3_ensemble,
            metadata=[],
        )

from esmda4d.production import (
    ProductionData,
    build_production_observations,
    build_production_ensemble,
)


def test_sr3_to_production_ensemble(
    sr3_ensemble,
):
    """
    Full production-data pipeline:

    ProductionData
        -> observation metadata

    SR3 ensemble
        -> CMGProductionReader
        -> SimulatedProductionData
        -> build_production_ensemble
        -> D
    """

    # ---------------------------------------------
    # 1. Define observations
    #
    # Notice the order:
    #
    # BHP Jan
    # BHP Mar
    # OIL Jan
    # OIL Mar
    #
    # D must follow exactly this order.
    # ---------------------------------------------

    observations = [
        ProductionData(
            entity="WELL-1",
            entity_type="well",
            variable="BHP",
            time=np.array([
                np.datetime64(
                    "2020-01-01"
                ),
                np.datetime64(
                    "2020-03-01"
                ),
            ]),
            values=np.array([
                105.0,
                125.0,
            ]),
            std=np.array([
                5.0,
                5.0,
            ]),
        ),
        ProductionData(
            entity="WELL-1",
            entity_type="well",
            variable="OILRATSC",
            time=np.array([
                np.datetime64(
                    "2020-01-01"
                ),
                np.datetime64(
                    "2020-03-01"
                ),
            ]),
            values=np.array([
                13.0,
                14.0,
            ]),
            std=np.array([
                2.0,
                2.0,
            ]),
        ),
    ]

    # ---------------------------------------------
    # 2. Build observations
    # ---------------------------------------------

    d_obs, Ce, metadata = (
        build_production_observations(
            observations
        )
    )

    # ---------------------------------------------
    # 3. Read corresponding quantities from
    #    every SR3 realization
    # ---------------------------------------------

    reader = CMGProductionReader()

    simulated = reader.read_ensemble(
        sr3_paths=sr3_ensemble,
        metadata=metadata,
    )

    # ---------------------------------------------
    # 4. Build ES-MDA predicted-data matrix
    # ---------------------------------------------

    D = build_production_ensemble(
        simulated,
        metadata,
    )

    # ---------------------------------------------
    # 5. Check observations
    # ---------------------------------------------

    np.testing.assert_allclose(
        d_obs,
        [
            105.0,
            125.0,
            13.0,
            14.0,
        ],
    )

    np.testing.assert_allclose(
        np.diag(Ce),
        [
            25.0,
            25.0,
            4.0,
            4.0,
        ],
    )

    # ---------------------------------------------
    # 6. Check simulated matrix
    #
    #                 Real. 1    Real. 2
    #
    # BHP Jan           100        101
    # BHP Mar           120        121
    # OIL Jan            10         15
    # OIL Mar            12         17
    # ---------------------------------------------

    expected_D = np.array([
        [100.0, 101.0],
        [120.0, 121.0],
        [10.0, 15.0],
        [12.0, 17.0],
    ])

    np.testing.assert_allclose(
        D,
        expected_D,
    )

    assert D.shape == (
        len(d_obs),
        2,
    )