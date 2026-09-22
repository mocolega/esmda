import numpy as np
import pytest

from esmda4d.model import (
    GridProperty,
    build_model_ensemble,
)
from esmda4d.cmg.writer import CMGModelWriter


def test_write_ensemble_creates_files(tmp_path):
    """
    The writer should create one realization directory
    per ensemble member, containing the CMG model file
    and the complete property include file.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    # -------------------------------------------------
    # Full prior property
    #
    # Grid: 2 x 2 x 1
    # Ensemble: 2 realizations
    #
    # CMG cell ordering:
    #
    # cell 0 -> (0,0,0)
    # cell 1 -> (1,0,0)
    # cell 2 -> (0,1,0)
    # cell 3 -> (1,1,0)
    # -------------------------------------------------

    prior = np.zeros(
        (2, 2, 1, 2),
        dtype=float,
    )

    prior[:, :, :, 0] = np.array(
        [0.10, 0.20, 0.30, 0.40]
    ).reshape(
        (2, 2, 1),
        order="F",
    )

    prior[:, :, :, 1] = np.array(
        [0.11, 0.21, 0.31, 0.41]
    ).reshape(
        (2, 2, 1),
        order="F",
    )

    # Only cells 1 and 3 participate in ES-MDA.

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([1, 3]),
        values=np.array([
            [0.25, 0.26],
            [0.45, 0.46],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    priors = {
        "porosity": prior,
    }

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    paths = writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors=priors,
    )

    assert len(paths) == 2

    # -------------------------------------------------
    # Realization 1
    # -------------------------------------------------

    model_1 = (
        tmp_path
        / "ensemble"
        / "realization_0001"
        / "model_0001.dat"
    )

    porosity_1 = (
        tmp_path
        / "ensemble"
        / "realization_0001"
        / "props"
        / "porosity_0001.inc"
    )

    assert model_1.is_file()
    assert porosity_1.is_file()

    assert paths[0] == model_1

    model_text = model_1.read_text()

    assert (
        "INCLUDE 'props/porosity_0001.inc'"
        in model_text
    )

    assert "$$porosity" not in model_text

    # Cells 0 and 2 retain prior values.
    # Cells 1 and 3 contain updated ES-MDA values.
    #
    # Expected:
    #
    # cell 0 -> 0.10  prior
    # cell 1 -> 0.25  updated
    # cell 2 -> 0.30  prior
    # cell 3 -> 0.45  updated

    assert (
        porosity_1.read_text().strip()
        == "0.1 0.25 0.3 0.45"
    )

    # -------------------------------------------------
    # Realization 2
    # -------------------------------------------------

    model_2 = (
        tmp_path
        / "ensemble"
        / "realization_0002"
        / "model_0002.dat"
    )

    porosity_2 = (
        tmp_path
        / "ensemble"
        / "realization_0002"
        / "props"
        / "porosity_0002.inc"
    )

    assert model_2.is_file()
    assert porosity_2.is_file()

    assert paths[1] == model_2

    assert (
        porosity_2.read_text().strip()
        == "0.11 0.26 0.31 0.46"
    )


def test_write_property_compresses_repeated_values(
    tmp_path,
):
    """
    Consecutive exactly equal values should use
    CMG N*V compression.
    """

    path = tmp_path / "property.inc"

    values = np.array([
        0.2,
        0.2,
        0.2,
        0.3,
        0.4,
        0.4,
    ])

    CMGModelWriter._write_property(
        path=path,
        values=values,
    )

    assert (
        path.read_text().strip()
        == "3*0.2 0.3 2*0.4"
    )


def test_write_property_respects_tokens_per_line(
    tmp_path,
):
    """
    tokens_per_line refers to output tokens.

    A compressed expression such as 3*0.2 counts
    as one token.
    """

    path = tmp_path / "property.inc"

    values = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    CMGModelWriter._write_property(
        path=path,
        values=values,
        tokens_per_line=2,
    )

    lines = path.read_text().splitlines()

    assert lines == [
        "1 2",
        "3 4",
        "5",
    ]


def test_writer_reconstructs_full_grid_in_cmg_order(
    tmp_path,
):
    """
    Verify full-grid reconstruction using a 4 x 3 x 2
    reservoir grid.

    CMG ordering is:

        I fastest,
        then J,
        then K.

    Therefore internal zero-based cell IDs are:

        (0,0,0) -> 0
        (1,0,0) -> 1
        (0,1,0) -> 4
        (1,1,0) -> 5
        (0,0,1) -> 12
        (3,2,1) -> 23
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    NI = 4
    NJ = 3
    NK = 2
    Ne = 2

    # Each prior value equals its CMG cell ID.
    #
    # Realization 1:
    # 0, 1, 2, ..., 23
    #
    # Realization 2:
    # 100, 101, ..., 123

    prior = np.zeros(
        (NI, NJ, NK, Ne),
        dtype=float,
    )

    prior[:, :, :, 0] = (
        np.arange(24).reshape(
            (NI, NJ, NK),
            order="F",
        )
    )

    prior[:, :, :, 1] = (
        (100 + np.arange(24)).reshape(
            (NI, NJ, NK),
            order="F",
        )
    )

    # Assimilate only cells:
    #
    # 1  -> (1,0,0)
    # 5  -> (1,1,0)
    # 12 -> (0,0,1)
    # 23 -> (3,2,1)

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([
            1,
            5,
            12,
            23,
        ]),
        values=np.array([
            [1001.0, 2001.0],
            [1005.0, 2005.0],
            [1012.0, 2012.0],
            [1023.0, 2023.0],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors={
            "porosity": prior,
        },
    )

    property_path = (
        tmp_path
        / "ensemble"
        / "realization_0001"
        / "props"
        / "porosity_0001.inc"
    )

    # No repeated values exist, so there should
    # be no N*V compression in this example.
    #
    # Read all whitespace-separated tokens.

    tokens = (
        property_path
        .read_text()
        .split()
    )

    values = np.array(
        [float(token) for token in tokens]
    )

    # A complete 4 x 3 x 2 property must contain
    # exactly 24 values.

    assert len(values) == 24

    expected = np.arange(
        24,
        dtype=float,
    )

    # Only assimilated cells change.

    expected[1] = 1001.0
    expected[5] = 1005.0
    expected[12] = 1012.0
    expected[23] = 1023.0

    np.testing.assert_array_equal(
        values,
        expected,
    )


def test_missing_prior_property_raises_error(
    tmp_path,
):
    """
    Every property contained in M must have a
    corresponding complete prior property.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([0]),
        values=np.array([
            [0.2, 0.3],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    with pytest.raises(
        ValueError,
        match="Missing prior property",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors={},
        )


def test_wrong_prior_ensemble_size_raises_error(
    tmp_path,
):
    """
    The prior and M must contain the same number
    of realizations.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([0]),
        values=np.array([
            [0.2, 0.3],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    # M has Ne = 2, but prior has Ne = 3.

    prior = np.ones(
        (2, 2, 1, 3)
    )

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    with pytest.raises(
        ValueError,
        match="same number",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors={
                "porosity": prior,
            },
        )


def test_template_with_unknown_placeholder_raises_error(
    tmp_path,
):
    """
    A template placeholder without a corresponding
    model property should raise an error.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
        "*PERMI ALL\n"
        "INCLUDE '$$permeability'\n"
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([0]),
        values=np.array([
            [0.2, 0.3],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    prior = np.ones(
        (2, 2, 1, 2)
    )

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    with pytest.raises(
        ValueError,
        match="no corresponding model property",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors={
                "porosity": prior,
            },
        )


def test_model_property_without_template_placeholder_raises_error(
    tmp_path,
):
    """
    A model property without a corresponding
    template placeholder should raise an error.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([0]),
        values=np.array([
            [0.2, 0.3],
        ]),
    )

    permeability = GridProperty(
        variable="permeability",
        cell_ids=np.array([0]),
        values=np.array([
            [100.0, 110.0],
        ]),
    )

    M, metadata = build_model_ensemble(
        [
            porosity,
            permeability,
        ]
    )

    prior_porosity = np.ones(
        (2, 2, 1, 2)
    )

    prior_permeability = np.ones(
        (2, 2, 1, 2)
    )

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    with pytest.raises(
        ValueError,
        match="no corresponding template",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors={
                "porosity": prior_porosity,
                "permeability": (
                    prior_permeability
                ),
            },
        )


def _make_two_realization_writer_case(
    tmp_path,
):
    """
    Create a minimal two-realization writer case
    for testing realization IDs.
    """

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    prior = np.zeros(
        (2, 2, 1, 2),
        dtype=float,
    )

    prior[:, :, :, 0] = np.array(
        [0.10, 0.20, 0.30, 0.40]
    ).reshape(
        (2, 2, 1),
        order="F",
    )

    prior[:, :, :, 1] = np.array(
        [0.11, 0.21, 0.31, 0.41]
    ).reshape(
        (2, 2, 1),
        order="F",
    )

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([1, 3]),
        values=np.array([
            [0.25, 0.26],
            [0.45, 0.46],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    priors = {
        "porosity": prior,
    }

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    return writer, M, metadata, priors

def test_writer_preserves_realization_ids(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    realization_ids = [
        1,
        3,
    ]

    paths = writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors=priors,
        realization_ids=realization_ids,
    )

    assert len(paths) == 2

    assert (
        paths[0].parent.name
        == "realization_0001"
    )

    assert (
        paths[0].name
        == "model_0001.dat"
    )

    assert (
        paths[1].parent.name
        == "realization_0003"
    )

    assert (
        paths[1].name
        == "model_0003.dat"
    )


def test_writer_rejects_wrong_number_of_ids(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="one ID",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors=priors,
            realization_ids=[1],
        )


def test_writer_rejects_duplicate_ids(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors=priors,
            realization_ids=[
                1,
                1,
            ],
        )


def test_writer_rejects_nonpositive_ids(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        writer.write_ensemble(
            M=M,
            metadata=metadata,
            priors=priors,
            realization_ids=[
                0,
                2,
            ],
        )


def test_writer_default_realization_ids(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    paths = writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors=priors,
    )

    assert (
        paths[0].parent.name
        == "realization_0001"
    )

    assert (
        paths[1].parent.name
        == "realization_0002"
    )

def test_writer_uses_realization_id_in_property_files(
    tmp_path,
):
    (
        writer,
        M,
        metadata,
        priors,
    ) = _make_two_realization_writer_case(
        tmp_path
    )

    paths = writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors=priors,
        realization_ids=[
            7,
            12,
        ],
    )

    property_7 = (
        tmp_path
        / "ensemble"
        / "realization_0007"
        / "props"
        / "porosity_0007.inc"
    )

    property_12 = (
        tmp_path
        / "ensemble"
        / "realization_0012"
        / "props"
        / "porosity_0012.inc"
    )

    assert property_7.is_file()
    assert property_12.is_file()

    assert (
        "INCLUDE 'props/porosity_0007.inc'"
        in paths[0].read_text()
    )

    assert (
        "INCLUDE 'props/porosity_0012.inc'"
        in paths[1].read_text()
    )