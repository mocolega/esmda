from pathlib import Path

import numpy as np

from esmda4d.grid import select_grid_property

def read_grid_include(
    path,
    ni,
    nj,
    nk,
):
    """
    Read a CMG grid-property include file.

    The file must contain exactly ni * nj * nk
    values after expanding CMG N*V notation.

    CMG grid ordering is used:
    I varies fastest, then J, then K.

    Parameters
    ----------
    path : str or Path
        Path to the CMG include file.

    ni, nj, nk : int
        Reservoir grid dimensions.

    Returns
    -------
    ndarray
        Grid values with shape (ni, nj, nk).
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"CMG include file not found: {path}"
        )

    for name, value in (
        ("ni", ni),
        ("nj", nj),
        ("nk", nk),
    ):
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 1:
            raise ValueError(
                f"{name} must be positive."
            )

    tokens = []

    for line in path.read_text().splitlines():

        # In CMG files, everything after "**"
        # is a comment.
        data = line.split("**", maxsplit=1)[0]

        tokens.extend(data.split())

    values = []

    for token in tokens:

        if "*" in token:
            parts = token.split("*")

            if len(parts) != 2:
                raise ValueError(
                    "Invalid CMG repeated-value token: "
                    f"{token}"
                )

            count_text, value_text = parts

            try:
                count = int(count_text)
                value = float(value_text)
            except ValueError as error:
                raise ValueError(
                    "Invalid CMG repeated-value token: "
                    f"{token}"
                ) from error

            if count < 1:
                raise ValueError(
                    "CMG repeated-value count must be "
                    f"positive: {token}"
                )

            values.extend(
                [value] * count
            )

        else:
            try:
                values.append(float(token))
            except ValueError as error:
                raise ValueError(
                    "Invalid numeric value in CMG "
                    f"include file: {token}"
                ) from error

    expected_size = ni * nj * nk

    if len(values) != expected_size:
        raise ValueError(
            f"CMG include file contains {len(values)} "
            f"grid values, but {expected_size} were "
            f"expected for a "
            f"{ni} x {nj} x {nk} grid."
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "CMG include file contains NaN or "
            "infinite values."
        )

    return values.reshape(
        (ni, nj, nk),
        order="F",
    )


def read_binary_grid_include(
    path,
    ni,
    nj,
    nk,
):
    """
    Read a CMG include file representing a binary mask.

    Values must contain only 0 and 1.

    Returns
    -------
    ndarray
        Boolean array with shape (ni, nj, nk).
    """

    values = read_grid_include(
        path=path,
        ni=ni,
        nj=nj,
        nk=nk,
    )

    if not np.all(
        np.logical_or(
            values == 0,
            values == 1,
        )
    ):
        raise ValueError(
            f"CMG mask file must contain only "
            f"0 and 1 values: {path}"
        )

    return values.astype(bool)


def read_prior_ensemble(
    paths,
    ni,
    nj,
    nk,
    n_realizations,
):
    """
    Read one CMG grid-property prior ensemble.

    Parameters
    ----------
    paths : sequence of str or Path
        One CMG include file per realization.

    ni, nj, nk : int
        Reservoir grid dimensions.

    n_realizations : int
        Expected number of realizations.

    Returns
    -------
    ndarray
        Prior ensemble with shape
        (ni, nj, nk, n_realizations).
    """

    paths = list(paths)

    if not isinstance(n_realizations, int):
        raise TypeError(
            "n_realizations must be an integer."
        )

    if n_realizations < 2:
        raise ValueError(
            "n_realizations must be at least 2."
        )

    if len(paths) != n_realizations:
        raise ValueError(
            f"Expected {n_realizations} prior files, "
            f"but {len(paths)} were provided."
        )

    realizations = []

    for path in paths:
        values = read_grid_include(
            path=path,
            ni=ni,
            nj=nj,
            nk=nk,
        )

        realizations.append(values)

    return np.stack(
        realizations,
        axis=-1,
    )

def load_grid_property(
    variable,
    prior_paths,
    null_mask,
    ni,
    nj,
    nk,
    n_realizations,
    property_mask_path=None,
):
    """
    Load one CMG grid property for ES-MDA.

    Returns the reduced GridProperty used by ES-MDA
    and the complete prior ensemble used for later
    grid reconstruction.
    """

    prior = read_prior_ensemble(
        paths=prior_paths,
        ni=ni,
        nj=nj,
        nk=nk,
        n_realizations=n_realizations,
    )

    if property_mask_path is None:
        property_mask = np.ones(
            (ni, nj, nk),
            dtype=bool,
        )
    else:
        property_mask = read_binary_grid_include(
            path=property_mask_path,
            ni=ni,
            nj=nj,
            nk=nk,
        )

    grid_property = select_grid_property(
        variable=variable,
        prior=prior,
        null_mask=null_mask,
        property_mask=property_mask,
    )

    return grid_property, prior


def load_grid_properties(
    properties,
    null_file,
    ni,
    nj,
    nk,
    n_realizations,
):
    """
    Load all CMG grid properties configured for ES-MDA.

    Returns
    -------
    grid_properties : list of GridProperty
        Reduced properties containing only cells that
        participate in history matching.

    priors : dict
        Complete prior ensembles keyed by variable name.
        Each array has shape
        (ni, nj, nk, n_realizations).
    """

    null_mask = read_binary_grid_include(
        path=null_file,
        ni=ni,
        nj=nj,
        nk=nk,
    )

    grid_properties = []
    priors = {}

    for variable, property_config in properties.items():
        grid_property, prior = load_grid_property(
            variable=variable,
            prior_paths=property_config["prior_files"],
            null_mask=null_mask,
            ni=ni,
            nj=nj,
            nk=nk,
            n_realizations=n_realizations,
            property_mask_path=property_config.get(
                "property_mask"
            ),
        )

        grid_properties.append(grid_property)
        priors[variable] = prior

    return grid_properties, priors