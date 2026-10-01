from .model import build_model_ensemble
from .cmg.grid import load_grid_properties


def build_initial_model(config):
    """
    Build the initial ES-MDA model ensemble from a
    resolved configuration.

    Parameters
    ----------
    config : dict
        Validated configuration with resolved file paths.

    Returns
    -------
    M : ndarray
        Model ensemble with shape
        (n_parameters, n_realizations).

    metadata : list
        Metadata describing each row of M.

    priors : dict
        Complete prior grid ensembles used later for
        reconstruction.
    """

    forward_model_type = config[
        "forward_model"
    ]["type"]

    if forward_model_type != "cmg":
        raise ValueError(
            "Unsupported forward model type: "
            f"{forward_model_type}"
        )

    grid_config = config["grid"]
    esmda_config = config["esmda"]

    grid_properties, priors = load_grid_properties(
        properties=config["properties"],
        null_file=grid_config["null_file"],
        ni=grid_config["ni"],
        nj=grid_config["nj"],
        nk=grid_config["nk"],
        n_realizations=esmda_config[
            "n_realizations"
        ],
    )

    M, metadata = build_model_ensemble(
        grid_properties
    )

    return M, metadata, priors