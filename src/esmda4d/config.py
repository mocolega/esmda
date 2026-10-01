import json
from pathlib import Path

from .inflation import validate_alphas

_REQUIRED_SECTIONS = {
    "esmda",
    "run",
    "forward_model",
    "grid",
    "properties",
    "observations",
}


def read_config(path):
    """
    Read the ES-MDA configuration from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the file does not contain valid JSON or if
        required top-level sections are missing.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    try:
        with path.open("r") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON configuration: {path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a JSON object "
            "at the top level."
        )

    missing = _REQUIRED_SECTIONS - config.keys()

    if missing:
        raise ValueError(
            "Configuration is missing required section(s): "
            + ", ".join(sorted(missing))
        )

    validate_esmda_config(
       config["esmda"]
    )
    validate_grid_config(
        config["grid"]
    )

    validate_properties_config(
        config["properties"],
        n_realizations=config["esmda"]["n_realizations"],
    )

    validate_forward_model_config(
        config["forward_model"]
    )

    validate_observations_config(
        config["observations"]
    )

    validate_run_config(
        config["run"]
    )

    validate_observations_config(
        config["observations"]
    )
    
    return config

def validate_esmda_config(config):
    """
    Validate the ES-MDA section of the configuration.

    Parameters
    ----------
    config : dict
        ES-MDA configuration section.

    Returns
    -------
    dict
        Validated ES-MDA configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'esmda' configuration must be an object."
        )

    required = {
        "n_realizations",
        "n_assimilations",
        "alphas",
        "inversion",
        "energy",
    }

    missing = required - config.keys()

    if missing:
        raise ValueError(
            "'esmda' configuration is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    n_realizations = config["n_realizations"]

    if (
        not isinstance(n_realizations, int)
        or isinstance(n_realizations, bool)
        or n_realizations < 2
    ):
        raise ValueError(
            "'n_realizations' must be an integer >= 2."
        )

    n_assimilations = config["n_assimilations"]

    if (
        not isinstance(n_assimilations, int)
        or isinstance(n_assimilations, bool)
        or n_assimilations < 1
    ):
        raise ValueError(
            "'n_assimilations' must be an integer >= 1."
        )

    alphas = validate_alphas(
        config["alphas"]
    )

    if len(alphas) != n_assimilations:
        raise ValueError(
            "Number of alphas must equal "
            "'n_assimilations'."
        )

    inversion = config["inversion"]

    if inversion not in {
        "direct",
        "tsvd",
    }:
        raise ValueError(
            "'inversion' must be 'direct' or 'tsvd'."
        )

    energy = config["energy"]

    if (
        isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not 0.0 < energy <= 1.0
    ):
        raise ValueError(
            "'energy' must be greater than 0 "
            "and less than or equal to 1."
        )

    return config

def validate_grid_config(config):
    """
    Validate the grid section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'grid' configuration must be an object."
        )

    required = {
        "ni",
        "nj",
        "nk",
        "null_file",
    }

    missing = required - config.keys()

    if missing:
        raise ValueError(
            "'grid' configuration is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    for name in ("ni", "nj", "nk"):
        value = config[name]

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(
                f"'{name}' must be a positive integer."
            )

    null_file = config["null_file"]

    if (
        not isinstance(null_file, str)
        or not null_file.strip()
    ):
        raise ValueError(
            "'null_file' must be a non-empty string."
        )

    return config


def validate_properties_config(
    config,
    n_realizations,
):
    """
    Validate the properties section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'properties' configuration must be an object."
        )

    if not config:
        raise ValueError(
            "'properties' must contain at least one property."
        )

    for variable, property_config in config.items():
        if (
            not isinstance(variable, str)
            or not variable.strip()
        ):
            raise ValueError(
                "Property names must be non-empty strings."
            )

        if not isinstance(property_config, dict):
            raise ValueError(
                f"Configuration for property '{variable}' "
                "must be an object."
            )

        if "prior_files" not in property_config:
            raise ValueError(
                f"Property '{variable}' is missing "
                "'prior_files'."
            )

        prior_files = property_config["prior_files"]

        if not isinstance(prior_files, list):
            raise ValueError(
                f"'prior_files' for property '{variable}' "
                "must be a list."
            )

        if len(prior_files) != n_realizations:
            raise ValueError(
                f"Property '{variable}' must contain exactly "
                f"{n_realizations} prior files."
            )

        for prior_file in prior_files:
            if (
                not isinstance(prior_file, str)
                or not prior_file.strip()
            ):
                raise ValueError(
                    f"All prior files for property '{variable}' "
                    "must be non-empty strings."
                )

        property_mask = property_config.get(
            "property_mask"
        )

        if (
            property_mask is not None
            and (
                not isinstance(property_mask, str)
                or not property_mask.strip()
            )
        ):
            raise ValueError(
                f"'property_mask' for property '{variable}' "
                "must be a non-empty string or null."
            )

    return config


def validate_forward_model_config(config):
    """
    Validate the forward-model section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'forward_model' configuration must be an object."
        )

    required = {
        "type",
        "template",
        "executable",
        "options",
        "max_simultaneous_runs",
        "max_retries",
        "delete_after_round",
    }

    missing = required - config.keys()

    if missing:
        raise ValueError(
            "'forward_model' configuration is missing "
            "required field(s): "
            + ", ".join(sorted(missing))
        )

    model_type = config["type"]

    if model_type != "cmg":
        raise ValueError(
            "'forward_model.type' must currently be 'cmg'."
        )

    for name in (
        "template",
        "executable",
    ):
        value = config[name]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"'forward_model.{name}' must be a "
                "non-empty string."
            )

    options = config["options"]

    if not isinstance(options, str):
        raise ValueError(
            "'forward_model.options' must be a string."
        )

    max_simultaneous_runs = config[
        "max_simultaneous_runs"
    ]

    if (
        not isinstance(max_simultaneous_runs, int)
        or isinstance(max_simultaneous_runs, bool)
        or max_simultaneous_runs < 1
    ):
        raise ValueError(
            "'forward_model.max_simultaneous_runs' "
            "must be a positive integer."
        )

    max_retries = config["max_retries"]

    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
    ):
        raise ValueError(
            "'forward_model.max_retries' must be "
            "an integer >= 0."
        )


    delete_after_round = config["delete_after_round"]

    if not isinstance(delete_after_round, list):
        raise ValueError(
            "'forward_model.delete_after_round' "
            "must be a list."
        )

    for pattern in delete_after_round:
        if (
            not isinstance(pattern, str)
            or not pattern.strip()
        ):
            raise ValueError(
                "'forward_model.delete_after_round' "
                "must contain only non-empty strings."
            )

    return config


def validate_observations_config(config):
    """
    Validate the observations section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'observations' configuration must be an object."
        )

    if "file" not in config:
        raise ValueError(
            "'observations' configuration is missing "
            "required field: file"
        )

    observation_file = config["file"]

    if (
        not isinstance(observation_file, str)
        or not observation_file.strip()
    ):
        raise ValueError(
            "'observations.file' must be a non-empty string."
        )

    return config

def validate_run_config(config):
    """
    Validate the run section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'run' configuration must be an object."
        )

    if "directory" not in config:
        raise ValueError(
            "'run' configuration is missing required "
            "field: directory"
        )

    directory = config["directory"]

    if (
        not isinstance(directory, str)
        or not directory.strip()
    ):
        raise ValueError(
            "'run.directory' must be a non-empty string."
        )

    return config

def validate_observations_config(config):
    """
    Validate the observations section of the configuration.
    """

    if not isinstance(config, dict):
        raise ValueError(
            "'observations' configuration must be an object."
        )

    if "file" not in config:
        raise ValueError(
            "'observations' configuration is missing "
            "required field: file"
        )

    observation_file = config["file"]

    if (
        not isinstance(observation_file, str)
        or not observation_file.strip()
    ):
        raise ValueError(
            "'observations.file' must be a non-empty string."
        )

    return config


def resolve_config_paths(config, config_path):
    """
    Resolve configuration file paths relative to the
    directory containing config.json.

    The forward-model executable is intentionally not
    resolved because it may be available through PATH.
    """

    config_path = Path(config_path)
    base_directory = config_path.parent.resolve()

    resolved = {
        section: (
            value.copy()
            if isinstance(value, dict)
            else value
        )
        for section, value in config.items()
    }

    # Run directory.
    resolved["run"]["directory"] = str(
        (
            base_directory
            / config["run"]["directory"]
        ).resolve()
    )

    # Forward-model template.
    resolved["forward_model"]["template"] = str(
        (
            base_directory
            / config["forward_model"]["template"]
        ).resolve()
    )

    # Grid null file.
    resolved["grid"]["null_file"] = str(
        (
            base_directory
            / config["grid"]["null_file"]
        ).resolve()
    )

    # Observation file.
    resolved["observations"]["file"] = str(
        (
            base_directory
            / config["observations"]["file"]
        ).resolve()
    )

    # Property files.
    resolved["properties"] = {}

    for variable, property_config in config[
        "properties"
    ].items():
        resolved_property = property_config.copy()

        resolved_property["prior_files"] = [
            str(
                (
                    base_directory
                    / prior_file
                ).resolve()
            )
            for prior_file
            in property_config["prior_files"]
        ]

        property_mask = property_config.get(
            "property_mask"
        )

        if property_mask is not None:
            resolved_property["property_mask"] = str(
                (
                    base_directory
                    / property_mask
                ).resolve()
            )

        resolved["properties"][
            variable
        ] = resolved_property

    return resolved