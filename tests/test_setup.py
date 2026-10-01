import numpy as np

from esmda4d.setup import build_initial_model
import json

from esmda4d.config import (
    read_config,
    resolve_config_paths,
)

def test_build_initial_model_from_cmg_config(tmp_path):
    null_path = tmp_path / "null.inc"
    null_path.write_text(
        "1 1 0 1 1 1 0 1\n"
    )

    por_paths = []
    perm_paths = []

    for realization in range(3):
        por_path = (
            tmp_path
            / f"por_{realization + 1:03d}.inc"
        )

        por_path.write_text(
            " ".join(
                str(value)
                for value in (
                    np.arange(1, 9)
                    + realization * 10
                )
            )
        )

        por_paths.append(por_path)

        perm_path = (
            tmp_path
            / f"perm_{realization + 1:03d}.inc"
        )

        perm_path.write_text(
            " ".join(
                str(value)
                for value in (
                    np.arange(101, 109)
                    + realization * 100
                )
            )
        )

        perm_paths.append(perm_path)

    config = {
        "esmda": {
            "n_realizations": 3,
        },
        "forward_model": {
            "type": "cmg",
        },
        "grid": {
            "ni": 2,
            "nj": 2,
            "nk": 2,
            "null_file": null_path,
        },
        "properties": {
            "POR": {
                "prior_files": por_paths,
                "property_mask": None,
            },
            "PERM": {
                "prior_files": perm_paths,
                "property_mask": None,
            },
        },
    }

    M, metadata, priors = build_initial_model(
        config
    )

    # Six active cells for each of two properties.
    assert M.shape == (12, 3)

    np.testing.assert_array_equal(
        M[:6],
        np.array(
            [
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [4.0, 14.0, 24.0],
                [5.0, 15.0, 25.0],
                [6.0, 16.0, 26.0],
                [8.0, 18.0, 28.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        M[6:],
        np.array(
            [
                [101.0, 201.0, 301.0],
                [102.0, 202.0, 302.0],
                [104.0, 204.0, 304.0],
                [105.0, 205.0, 305.0],
                [106.0, 206.0, 306.0],
                [108.0, 208.0, 308.0],
            ]
        ),
    )

    assert len(metadata) == 12

    assert metadata[0] == {
        "variable": "POR",
        "cell_id": 0,
    }

    assert metadata[5] == {
        "variable": "POR",
        "cell_id": 7,
    }

    assert metadata[6] == {
        "variable": "PERM",
        "cell_id": 0,
    }

    assert metadata[-1] == {
        "variable": "PERM",
        "cell_id": 7,
    }

    assert set(priors) == {
        "POR",
        "PERM",
    }

    assert priors["POR"].shape == (
        2,
        2,
        2,
        3,
    )

    assert priors["PERM"].shape == (
        2,
        2,
        2,
        3,
    )

def test_build_initial_model_from_config_file(tmp_path):
    grid_dir = tmp_path / "grid"
    priors_dir = tmp_path / "priors"
    observations_dir = tmp_path / "observations"
    model_dir = tmp_path / "model"

    grid_dir.mkdir()
    priors_dir.mkdir()
    observations_dir.mkdir()
    model_dir.mkdir()

    # Global reservoir mask.
    (grid_dir / "null.inc").write_text(
        "1 1 0 1 1 1 0 1\n"
    )

    # Three POR realizations.
    for realization in range(3):
        values = (
            np.arange(1, 9)
            + realization * 10
        )

        (
            priors_dir
            / f"por_{realization + 1:03d}.inc"
        ).write_text(
            " ".join(
                str(value)
                for value in values
            )
        )

    config = {
        "esmda": {
            "n_realizations": 3,
            "n_assimilations": 2,
            "alphas": [2.0, 2.0],
            "inversion": "direct",
            "energy": 0.99,
        },
        "run": {
            "directory": "my_run_folder",
        },
        "forward_model": {
            "type": "cmg",
            "template": "model/template.dat",
            "executable": "mx2025.exe",
            "options": "",
            "max_simultaneous_runs": 2,
            "max_retries": 1,
            "delete_after_round": [],
        },
        "grid": {
            "ni": 2,
            "nj": 2,
            "nk": 2,
            "null_file": "grid/null.inc",
        },
        "properties": {
            "POR": {
                "prior_files": [
                    "priors/por_001.inc",
                    "priors/por_002.inc",
                    "priors/por_003.inc",
                ],
                "property_mask": None,
            },
        },
        "observations": {
            "file": "observations/production.json",
        },
    }

    config_path = tmp_path / "config.json"

    config_path.write_text(
        json.dumps(config)
    )

    config = read_config(
        config_path
    )

    config = resolve_config_paths(
        config,
        config_path,
    )

    M, metadata, priors = build_initial_model(
        config
    )

    assert M.shape == (6, 3)

    np.testing.assert_array_equal(
        M,
        np.array(
            [
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [4.0, 14.0, 24.0],
                [5.0, 15.0, 25.0],
                [6.0, 16.0, 26.0],
                [8.0, 18.0, 28.0],
            ]
        ),
    )

    assert metadata == [
        {"variable": "POR", "cell_id": 0},
        {"variable": "POR", "cell_id": 1},
        {"variable": "POR", "cell_id": 3},
        {"variable": "POR", "cell_id": 4},
        {"variable": "POR", "cell_id": 5},
        {"variable": "POR", "cell_id": 7},
    ]

    assert set(priors) == {"POR"}

    assert priors["POR"].shape == (
        2,
        2,
        2,
        3,
    )

    # Paths were resolved relative to config.json.
    assert config["grid"]["null_file"] == str(
        (grid_dir / "null.inc").resolve()
    )

    assert config["run"]["directory"] == str(
        (
            tmp_path
            / "my_run_folder"
        ).resolve()
    )

    # Executable was deliberately left unchanged.
    assert (
        config["forward_model"]["executable"]
        == "mx2025.exe"
    )