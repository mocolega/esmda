import sys

import numpy as np

from esmda4d.model import (
    GridProperty,
    build_model_ensemble,
)
from esmda4d.cmg.writer import CMGModelWriter
from esmda4d.cmg.runner import LocalCMGRunner


def test_writer_to_local_runner(tmp_path):
    """
    Integration test:

    GridProperty
        -> M
        -> CMGModelWriter
        -> model_XXXX.dat
        -> LocalCMGRunner

    Python itself is used as a fake simulator so the
    test does not require a CMG installation.
    """

    # -------------------------------------------------
    # 1. Create CMG template
    # -------------------------------------------------

    template_path = tmp_path / "model.tpl"

    template_path.write_text(
        "*POR ALL\n"
        "INCLUDE '$$porosity'\n"
    )

    # -------------------------------------------------
    # 2. Create full prior ensemble
    #
    # Grid:
    # NI = 2
    # NJ = 2
    # NK = 1
    #
    # Ne = 2
    # -------------------------------------------------

    prior = np.zeros(
        (2, 2, 1, 2),
        dtype=float,
    )

    prior[:, :, :, 0] = np.array([
        0.10,
        0.20,
        0.30,
        0.40,
    ]).reshape(
        (2, 2, 1),
        order="F",
    )

    prior[:, :, :, 1] = np.array([
        0.11,
        0.21,
        0.31,
        0.41,
    ]).reshape(
        (2, 2, 1),
        order="F",
    )

    # -------------------------------------------------
    # 3. ES-MDA property
    #
    # Only cells 1 and 3 are updated.
    # -------------------------------------------------

    porosity = GridProperty(
        variable="porosity",
        cell_ids=np.array([
            1,
            3,
        ]),
        values=np.array([
            [0.25, 0.26],
            [0.45, 0.46],
        ]),
    )

    M, metadata = build_model_ensemble(
        [porosity]
    )

    # -------------------------------------------------
    # 4. Write CMG realizations
    # -------------------------------------------------

    writer = CMGModelWriter(
        template_path=template_path,
        output_dir=tmp_path / "ensemble",
    )

    model_paths = writer.write_ensemble(
        M=M,
        metadata=metadata,
        priors={
            "porosity": prior,
        },
    )

    assert len(model_paths) == 2

    # -------------------------------------------------
    # 5. Create a fake simulator
    #
    # The script:
    #
    # - receives the model filename
    # - checks that it exists
    # - reads it
    # - prints its name
    #
    # Because LocalCMGRunner uses the realization
    # directory as cwd, the model filename should
    # resolve correctly.
    # -------------------------------------------------

    fake_simulator = (
        tmp_path / "fake_simulator.py"
    )

    fake_simulator.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "model = Path(sys.argv[1])\n"
        "\n"
        "if not model.is_file():\n"
        "    sys.exit(1)\n"
        "\n"
        "text = model.read_text()\n"
        "\n"
        "if 'INCLUDE' not in text:\n"
        "    sys.exit(2)\n"
        "\n"
        "print(model.name)\n"
    )

    # -------------------------------------------------
    # 6. Run generated models
    #
    # sys.executable is the current Python executable.
    #
    # Command becomes approximately:
    #
    # python fake_simulator.py model_0001.dat
    #
    # However, fake_simulator.py is outside the
    # realization directory, so we pass its absolute
    # path.
    # -------------------------------------------------

    runner = LocalCMGRunner(
        executable=sys.executable,
        arguments=[
            str(fake_simulator),
            "{model}",
        ],
    )

    results = runner.run_ensemble(
        model_paths
    )

    # -------------------------------------------------
    # 7. Verify execution
    # -------------------------------------------------

    assert len(results) == 2

    assert results[0].succeeded
    assert results[1].succeeded

    assert results[0].return_code == 0
    assert results[1].return_code == 0

    assert (
        "model_0001.dat"
        in results[0].stdout
    )

    assert (
        "model_0002.dat"
        in results[1].stdout
    )

    # -------------------------------------------------
    # 8. Verify paths are preserved
    # -------------------------------------------------

    assert (
        results[0].model_path
        == model_paths[0]
    )

    assert (
        results[1].model_path
        == model_paths[1]
    )