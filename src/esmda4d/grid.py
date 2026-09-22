import numpy as np

from .model import GridProperty


def select_grid_property(
    variable,
    prior,
    null_mask,
    property_mask,
):
    """
    Select the cells of a full-grid property that will
    participate in data assimilation.

    Parameters
    ----------
    variable : str
        Property name.

    prior : ndarray
        Prior property ensemble with shape
        (NI, NJ, NK, Ne).

    null_mask : ndarray
        Boolean array with shape (NI, NJ, NK).
        True identifies valid reservoir cells.

    property_mask : ndarray
        Boolean array with shape (NI, NJ, NK).
        True identifies cells selected by the user
        for data assimilation.

    Returns
    -------
    GridProperty
        Property containing only the cells that
        participate in data assimilation.
    """

    prior = np.asarray(
        prior,
        dtype=float,
    )

    null_mask = np.asarray(
        null_mask,
        dtype=bool,
    )

    property_mask = np.asarray(
        property_mask,
        dtype=bool,
    )

    if prior.ndim != 4:
        raise ValueError(
            "prior must have shape "
            "(NI, NJ, NK, n_ensemble)."
        )

    grid_shape = prior.shape[:3]

    if null_mask.shape != grid_shape:
        raise ValueError(
            "null_mask must have the same grid "
            "shape as prior."
        )

    if property_mask.shape != grid_shape:
        raise ValueError(
            "property_mask must have the same grid "
            "shape as prior."
        )

    effective_mask = (
        null_mask
        & property_mask
    )

    flat_mask = effective_mask.ravel(order="F")

    cell_ids = np.flatnonzero(
        flat_mask
    )

    if len(cell_ids) == 0:
        raise ValueError(
            f"No cells selected for property "
            f"{variable}."
        )

    Ne = prior.shape[3]

    flat_prior = np.column_stack([
        prior[:, :, :, j].ravel(order="F")
        for j in range(Ne)
    ])

    values = flat_prior[
        cell_ids,
        :
    ]

    return GridProperty(
        variable=variable,
        cell_ids=cell_ids,
        values=values,
    )



def reconstruct_grid_property(
    grid_property,
    prior,
):
    """
    Reconstruct a full-grid property ensemble from
    updated data-assimilation cells and the prior.

    Cells outside the data-assimilation region retain
    their prior values.

    CMG ordering is used:
    I varies fastest, then J, then K.

    Parameters
    ----------
    grid_property : GridProperty
        Updated property containing only the cells
        participating in data assimilation.

    prior : ndarray
        Full prior property ensemble with shape
        (NI, NJ, NK, Ne).

    Returns
    -------
    ndarray
        Full reconstructed property ensemble with
        shape (NI, NJ, NK, Ne).
    """

    if not isinstance(
        grid_property,
        GridProperty,
    ):
        raise TypeError(
            "grid_property must be a "
            "GridProperty object."
        )

    prior = np.asarray(
        prior,
        dtype=float,
    )

    if prior.ndim != 4:
        raise ValueError(
            "prior must have shape "
            "(NI, NJ, NK, n_ensemble)."
        )

    Ne = prior.shape[3]

    if grid_property.values.shape[1] != Ne:
        raise ValueError(
            "GridProperty and prior must contain "
            "the same number of ensemble "
            "realizations."
        )

    n_cells = np.prod(
        prior.shape[:3]
    )

    if (
        np.any(grid_property.cell_ids < 0)
        or np.any(
            grid_property.cell_ids >= n_cells
        )
    ):
        raise ValueError(
            "GridProperty contains cell IDs outside "
            "the reservoir grid."
        )

    full_property = prior.copy()

    for j in range(Ne):

        # Flatten one realization using CMG ordering:
        # I varies fastest, then J, then K.

        flat_property = (
            full_property[:, :, :, j]
            .ravel(order="F")
            .copy()
        )

        # Replace only cells participating in
        # data assimilation.

        flat_property[
            grid_property.cell_ids
        ] = grid_property.values[:, j]

        # Convert back to the 3D reservoir grid.

        full_property[:, :, :, j] = (
            flat_property.reshape(
                prior.shape[:3],
                order="F",
            )
        )

    return full_property