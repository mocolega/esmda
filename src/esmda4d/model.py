from dataclasses import dataclass
import numpy as np


@dataclass
class GridProperty:
    """
    Reservoir grid property across an ensemble.

    Parameters
    ----------
    variable : str
        Name of the reservoir property, such as
        "permeability", "porosity", or "ntg".

    cell_ids : ndarray
        Identifiers of the grid cells represented
        by the property.

    values : ndarray
        Property values with shape
        (n_cells, n_ensemble).

        Rows correspond to grid cells and columns
        correspond to ensemble realizations.
    """

    variable: str
    cell_ids: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        self.cell_ids = np.asarray(
            self.cell_ids
        )

        self.values = np.asarray(
            self.values,
            dtype=float,
        )

        self._validate()

    def _validate(self):

        if self.cell_ids.ndim != 1:
            raise ValueError(
                "cell_ids must be a 1D array."
            )

        if self.values.ndim != 2:
            raise ValueError(
                "values must be a 2D array with shape "
                "(n_cells, n_ensemble)."
            )

        if len(self.cell_ids) != self.values.shape[0]:
            raise ValueError(
                "The number of cell IDs must match "
                "the number of rows in values."
            )

        if len(self.cell_ids) == 0:
            raise ValueError(
                "GridProperty cannot be empty."
            )

        if self.values.shape[1] < 2:
            raise ValueError(
                "The ensemble must contain at least "
                "2 realizations."
            )

        if not np.all(
            np.isfinite(self.values)
        ):
            raise ValueError(
                "values contains NaN or infinite values."
            )

def build_model_ensemble(grid_properties):
    """
    Build the ES-MDA model ensemble matrix and metadata
    from GridProperty objects.

    Parameters
    ----------
    grid_properties : sequence of GridProperty
        Reservoir grid properties to include in the
        ES-MDA model vector.

    Returns
    -------
    M : ndarray
        Model ensemble matrix with shape
        (n_model_parameters, n_ensemble).

        Rows correspond to model parameters and columns
        correspond to ensemble realizations.

    metadata : list of dict
        Metadata describing each row of M.
    """

    if len(grid_properties) == 0:
        raise ValueError(
            "grid_properties cannot be empty."
        )

    for data in grid_properties:
        if not isinstance(data, GridProperty):
            raise TypeError(
                "All elements of grid_properties must "
                "be GridProperty objects."
            )

    # All properties must contain the same number
    # of ensemble realizations.

    Ne = grid_properties[0].values.shape[1]

    for data in grid_properties:
        if data.values.shape[1] != Ne:
            raise ValueError(
                "All grid properties must contain "
                "the same number of ensemble "
                "realizations."
            )

    rows = []
    metadata = []

    for data in grid_properties:

        rows.append(data.values)

        for cell_id in data.cell_ids:
            metadata.append({
                "variable": data.variable,
                "cell_id": cell_id,
            })

    M = np.vstack(rows)

    return M, metadata

def unpack_model_ensemble(M, metadata):
    """
    Convert an ES-MDA model ensemble matrix back into
    GridProperty objects.

    Parameters
    ----------
    M : ndarray
        Model ensemble matrix with shape
        (n_model_parameters, n_ensemble).

    metadata : sequence of dict
        Metadata describing each row of M, as generated
        by build_model_ensemble().

    Returns
    -------
    grid_properties : list of GridProperty
        Grid properties reconstructed from M.
    """

    M = np.asarray(
        M,
        dtype=float,
    )

    if M.ndim != 2:
        raise ValueError(
            "M must be a 2D array with shape "
            "(n_model_parameters, n_ensemble)."
        )

    if M.shape[0] != len(metadata):
        raise ValueError(
            "The number of rows in M must match "
            "the number of metadata entries."
        )

    if M.shape[0] == 0:
        raise ValueError(
            "M cannot be empty."
        )

    if M.shape[1] < 2:
        raise ValueError(
            "The ensemble must contain at least "
            "2 realizations."
        )

    if not np.all(np.isfinite(M)):
        raise ValueError(
            "M contains NaN or infinite values."
        )

    # Preserve the order in which variables first
    # appear in metadata.

    variables = []

    for item in metadata:
        variable = item["variable"]

        if variable not in variables:
            variables.append(variable)

    grid_properties = []

    for variable in variables:

        indices = [
            i
            for i, item in enumerate(metadata)
            if item["variable"] == variable
        ]

        cell_ids = np.array([
            metadata[i]["cell_id"]
            for i in indices
        ])

        values = M[indices, :]

        grid_properties.append(
            GridProperty(
                variable=variable,
                cell_ids=cell_ids,
                values=values,
            )
        )

    return grid_properties