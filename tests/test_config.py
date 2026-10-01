import json

import pytest

from esmda4d.config import (
    read_config,
    resolve_config_paths,
)



def test_read_config(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()

    path.write_text(
        json.dumps(config)
    )

    result = read_config(path)

    assert result == config


def test_read_config_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        read_config(path)


def test_read_config_rejects_invalid_json(tmp_path):
    path = tmp_path / "config.json"

    path.write_text(
        "{invalid json"
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON configuration",
    ):
        read_config(path)


def test_read_config_rejects_non_object(tmp_path):
    path = tmp_path / "config.json"

    path.write_text(
        json.dumps(["esmda", "grid"])
    )

    with pytest.raises(
        ValueError,
        match="top level",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "section",
    [
        "esmda",
        "run",
        "forward_model",
        "grid",
        "properties",
        "observations",
    ],
)
def test_read_config_rejects_missing_section(
    tmp_path,
    section,
):
    path = tmp_path / "config.json"

    config = valid_config()
    del config[section]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match=section,
    ):
        read_config(path)

def valid_config():
    return {
        "esmda": {
            "n_realizations": 3,
            "n_assimilations": 4,
            "alphas": [4.0, 4.0, 4.0, 4.0],
            "inversion": "direct",
            "energy": 0.99,
        },
        "forward_model": {
            "type": "cmg",
            "template": "model/template.dat",
            "executable": "mx2025.exe",
            "options": "",
            "max_simultaneous_runs": 4,
            "max_retries": 1,
            "delete_after_round": [
               "*.out",
               "*.irf",
               "*.mrf",
            ],
        },
        "grid": {
            "ni": 139,
            "nj": 48,
            "nk": 9,
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
        "run": {
          "directory": "my_run_folder",
        },
    }

def test_read_config_rejects_missing_esmda_field(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["esmda"]["n_realizations"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="n_realizations",
    ):
        read_config(path)


def test_read_config_rejects_invalid_n_realizations(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["esmda"]["n_realizations"] = 1

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="n_realizations",
    ):
        read_config(path)


def test_read_config_rejects_invalid_n_assimilations(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["esmda"]["n_assimilations"] = 0

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="n_assimilations",
    ):
        read_config(path)


def test_read_config_rejects_wrong_number_of_alphas(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["esmda"]["alphas"] = [2.0, 2.0]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="Number of alphas",
    ):
        read_config(path)


def test_read_config_rejects_invalid_inversion(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["esmda"]["inversion"] = "invalid"

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="inversion",
    ):
        read_config(path)


def test_read_config_rejects_invalid_energy(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["esmda"]["energy"] = 1.5

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="energy",
    ):
        read_config(path)

def test_read_config_rejects_missing_grid_field(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["grid"]["null_file"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="null_file",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ni", 0),
        ("nj", -1),
        ("nk", 1.5),
        ("ni", True),
    ],
)
def test_read_config_rejects_invalid_grid_dimension(
    tmp_path,
    field,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["grid"][field] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match=field,
    ):
        read_config(path)


def test_read_config_rejects_invalid_null_file(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["grid"]["null_file"] = ""

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="null_file",
    ):
        read_config(path)

def test_read_config_rejects_empty_properties(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["properties"] = {}

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="at least one property",
    ):
        read_config(path)


def test_read_config_rejects_missing_prior_files(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["properties"]["POR"]["prior_files"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="prior_files",
    ):
        read_config(path)


def test_read_config_rejects_wrong_number_of_prior_files(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()

    config["properties"]["POR"]["prior_files"] = [
        "priors/por_001.inc",
        "priors/por_002.inc",
    ]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="exactly 3 prior files",
    ):
        read_config(path)


def test_read_config_rejects_invalid_prior_file(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()

    config["properties"]["POR"]["prior_files"][1] = ""

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="prior files",
    ):
        read_config(path)


def test_read_config_accepts_null_property_mask(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()
    config["properties"]["POR"]["property_mask"] = None

    path.write_text(json.dumps(config))

    result = read_config(path)

    assert (
        result["properties"]["POR"]["property_mask"]
        is None
    )


def test_read_config_accepts_property_mask_file(tmp_path):
    path = tmp_path / "config.json"

    config = valid_config()

    config["properties"]["POR"][
        "property_mask"
    ] = "masks/por.inc"

    path.write_text(json.dumps(config))

    result = read_config(path)

    assert (
        result["properties"]["POR"]["property_mask"]
        == "masks/por.inc"
    )


def test_read_config_rejects_invalid_property_mask(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()

    config["properties"]["POR"]["property_mask"] = ""

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="property_mask",
    ):
        read_config(path)

def test_read_config_rejects_missing_forward_model_field(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["forward_model"]["executable"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="executable",
    ):
        read_config(path)


def test_read_config_rejects_invalid_forward_model_type(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"]["type"] = "unknown"

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="forward_model.type",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("template", ""),
        ("executable", ""),
    ],
)
def test_read_config_rejects_invalid_forward_model_path(
    tmp_path,
    field,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"][field] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match=field,
    ):
        read_config(path)


def test_read_config_rejects_invalid_forward_model_options(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"]["options"] = ["-foo"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="options",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_read_config_rejects_invalid_max_simultaneous_runs(
    tmp_path,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"][
        "max_simultaneous_runs"
    ] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="max_simultaneous_runs",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "value",
    [
        -1,
        1.5,
        True,
    ],
)
def test_read_config_rejects_invalid_max_retries(
    tmp_path,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"]["max_retries"] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="max_retries",
    ):
        read_config(path)

def test_read_config_rejects_missing_observation_file(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["observations"]["file"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="file",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_read_config_rejects_invalid_observation_file(
    tmp_path,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["observations"]["file"] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="observations.file",
    ):
        read_config(path)

def test_read_config_rejects_missing_run_directory(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    del config["run"]["directory"]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="directory",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_read_config_rejects_invalid_run_directory(
    tmp_path,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["run"]["directory"] = value

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="run.directory",
    ):
        read_config(path)


def test_read_config_accepts_empty_delete_after_round(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"]["delete_after_round"] = []

    path.write_text(json.dumps(config))

    result = read_config(path)

    assert (
        result["forward_model"]["delete_after_round"]
        == []
    )


def test_read_config_rejects_invalid_delete_after_round(
    tmp_path,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"][
        "delete_after_round"
    ] = "*.out"

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="delete_after_round",
    ):
        read_config(path)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_read_config_rejects_invalid_delete_pattern(
    tmp_path,
    value,
):
    path = tmp_path / "config.json"

    config = valid_config()
    config["forward_model"][
        "delete_after_round"
    ] = ["*.out", value]

    path.write_text(json.dumps(config))

    with pytest.raises(
        ValueError,
        match="delete_after_round",
    ):
        read_config(path)


def test_resolve_config_paths(tmp_path):
    config_directory = tmp_path / "case"
    config_directory.mkdir()

    config_path = config_directory / "config.json"

    config = valid_config()

    resolved = resolve_config_paths(
        config,
        config_path,
    )

    assert resolved["run"]["directory"] == str(
        (
            config_directory
            / "my_run_folder"
        ).resolve()
    )

    assert resolved["forward_model"]["template"] == str(
        (
            config_directory
            / "model/template.dat"
        ).resolve()
    )

    assert resolved["grid"]["null_file"] == str(
        (
            config_directory
            / "grid/null.inc"
        ).resolve()
    )

    assert resolved["observations"]["file"] == str(
        (
            config_directory
            / "observations/production.json"
        ).resolve()
    )

    assert resolved["properties"]["POR"][
        "prior_files"
    ] == [
        str(
            (
                config_directory
                / "priors/por_001.inc"
            ).resolve()
        ),
        str(
            (
                config_directory
                / "priors/por_002.inc"
            ).resolve()
        ),
        str(
            (
                config_directory
                / "priors/por_003.inc"
            ).resolve()
        ),
    ]


def test_resolve_config_paths_preserves_executable_and_original(
    tmp_path,
):
    config_path = tmp_path / "config.json"

    config = valid_config()

    config["properties"]["POR"][
        "property_mask"
    ] = "masks/por.inc"

    resolved = resolve_config_paths(
        config,
        config_path,
    )

    # Executable may come from the system PATH.
    assert (
        resolved["forward_model"]["executable"]
        == "mx2025.exe"
    )

    # Property mask is resolved.
    assert resolved["properties"]["POR"][
        "property_mask"
    ] == str(
        (
            tmp_path
            / "masks/por.inc"
        ).resolve()
    )

    # Original configuration is unchanged.
    assert (
        config["run"]["directory"]
        == "my_run_folder"
    )

    assert (
        config["forward_model"]["template"]
        == "model/template.dat"
    )

    assert (
        config["properties"]["POR"][
            "property_mask"
        ]
        == "masks/por.inc"
    )


