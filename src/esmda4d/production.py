from dataclasses import dataclass

import numpy as np


@dataclass
class ProductionData:
    """
    Production data for one entity and one variable.

    Parameters
    ----------
    entity : str
        Name of the entity associated with the data,
        for example a well, sector, group, or field.

    entity_type : str
        Type of the entity, for example:
        "well", "sector", "group", "field".

    variable : str
        Production variable, for example:
        "oil_rate", "water_rate", "bhp".

    time : ndarray
        Observation times with shape (n_times,).

    values : ndarray
        Production values with shape (n_times,).

    std : ndarray
        Observation-error standard deviations with
        shape (n_times,).
    """

    entity: str
    entity_type: str
    variable: str
    time: np.ndarray
    values: np.ndarray
    std: np.ndarray

    def __post_init__(self):

        self.time = np.asarray(self.time)
        self.values = np.asarray(
            self.values,
            dtype=float,
        )
        self.std = np.asarray(
            self.std,
            dtype=float,
        )

        self._validate()

    def _validate(self):

        if self.time.ndim != 1:
            raise ValueError(
                "time must be a 1D array."
            )

        if self.values.ndim != 1:
            raise ValueError(
                "values must be a 1D array."
            )

        if self.std.ndim != 1:
            raise ValueError(
                "std must be a 1D array."
            )

        if not (
            len(self.time)
            == len(self.values)
            == len(self.std)
        ):
            raise ValueError(
                "time, values, and std must have "
                "the same length."
            )

        if len(self.values) == 0:
            raise ValueError(
                "ProductionData cannot be empty."
            )

        if np.any(self.std <= 0):
            raise ValueError(
                "All observation-error standard "
                "deviations must be positive."
            )

        if not np.all(
            np.isfinite(self.values)
        ):
            raise ValueError(
                "values contains NaN or infinite values."
            )

        if not np.all(
            np.isfinite(self.std)
        ):
            raise ValueError(
                "std contains NaN or infinite values."
            )


@dataclass
class SimulatedProductionData:
    """
    Simulated production data for one entity and one variable
    across an ensemble of reservoir realizations.

    Parameters
    ----------
    entity : str
        Name of the entity associated with the data,
        for example a well, sector, group, or field.

    entity_type : str
        Type of entity, for example:
        "well", "sector", "group", or "field".

    variable : str
        Production variable, for example:
        "oil_rate", "water_rate", or "bhp".

    time : ndarray
        Simulation times with shape (n_times,).

    values : ndarray
        Simulated production values with shape
        (n_times, n_ensemble).

        Each row corresponds to one time and each
        column corresponds to one ensemble realization.
    """

    entity: str
    entity_type: str
    variable: str
    time: np.ndarray
    values: np.ndarray

    def __post_init__(self):

        self.time = np.asarray(self.time)

        self.values = np.asarray(
            self.values,
            dtype=float,
        )

        self._validate()

    def _validate(self):

        if self.time.ndim != 1:
            raise ValueError(
                "time must be a 1D array."
            )

        if self.values.ndim != 2:
            raise ValueError(
                "values must be a 2D array with shape "
                "(n_times, n_ensemble)."
            )

        if len(self.time) != self.values.shape[0]:
            raise ValueError(
                "The number of time points must match "
                "the number of rows in values."
            )

        if len(self.time) == 0:
            raise ValueError(
                "SimulatedProductionData cannot be empty."
            )

        if self.values.shape[1] < 2:
            raise ValueError(
                "The simulated ensemble must contain "
                "at least 2 realizations."
            )

        if not np.all(
            np.isfinite(self.values)
        ):
            raise ValueError(
                "values contains NaN or infinite values."
            )

def build_production_observations(production_data):
    """
    Build the ES-MDA observation vector, error covariance
    matrix, and metadata from ProductionData objects.

    Parameters
    ----------
    production_data : sequence of ProductionData
        Production data series to include in the
        assimilation.

    Returns
    -------
    d_obs : ndarray
        Observation vector with shape (n_data,).

    Ce : ndarray
        Observation-error covariance matrix with shape
        (n_data, n_data).

    metadata : list of dict
        Metadata associated with each element of d_obs.
    """

    if len(production_data) == 0:
        raise ValueError(
            "production_data cannot be empty."
        )

    values = []
    std = []
    metadata = []

    for data in production_data:

        if not isinstance(data, ProductionData):
            raise TypeError(
                "All elements of production_data must "
                "be ProductionData objects."
            )

        values.append(data.values)
        std.append(data.std)

        for time in data.time:

            metadata.append({
                "entity": data.entity,
                "entity_type": data.entity_type,
                "variable": data.variable,
                "time": time,
            })

    d_obs = np.concatenate(values)

    std = np.concatenate(std)

    Ce = np.diag(
        std**2
    )

    return d_obs, Ce, metadata


def build_production_ensemble(
    simulated_data,
    metadata,
):
    """
    Build the ES-MDA predicted-data ensemble matrix
    from simulated production data.

    Parameters
    ----------
    simulated_data : sequence of SimulatedProductionData
        Simulated production data for the ensemble.

    metadata : sequence of dict
        Observation metadata generated by
        build_production_observations().

    Returns
    -------
    D : ndarray
        Predicted-data ensemble matrix with shape
        (n_data, n_ensemble).

        Each row corresponds to one observation and
        each column corresponds to one ensemble
        realization.
    """

    if len(simulated_data) == 0:
        raise ValueError(
            "simulated_data cannot be empty."
        )

    if len(metadata) == 0:
        raise ValueError(
            "metadata cannot be empty."
        )

    for data in simulated_data:
        if not isinstance(
            data,
            SimulatedProductionData,
        ):
            raise TypeError(
                "All elements of simulated_data must "
                "be SimulatedProductionData objects."
            )

    # All simulated datasets must contain the same
    # number of ensemble realizations.

    Ne = simulated_data[0].values.shape[1]

    for data in simulated_data:
        if data.values.shape[1] != Ne:
            raise ValueError(
                "All simulated production data must "
                "contain the same number of ensemble "
                "realizations."
            )

    rows = []

    for obs in metadata:

        matches = [
            data
            for data in simulated_data
            if (
                data.entity == obs["entity"]
                and data.entity_type == obs["entity_type"]
                and data.variable == obs["variable"]
            )
        ]

        if len(matches) == 0:
            raise ValueError(
                "No simulated production data found for "
                f"entity={obs['entity']}, "
                f"entity_type={obs['entity_type']}, "
                f"variable={obs['variable']}."
            )

        if len(matches) > 1:
            raise ValueError(
                "Multiple simulated production datasets "
                "match the same observation."
            )

        data = matches[0]

        time_indices = np.where(
            data.time == obs["time"]
        )[0]

        if len(time_indices) == 0:
            raise ValueError(
                "Observation time not found in simulated "
                "production data. "
                f"entity={obs['entity']}, "
                f"variable={obs['variable']}, "
                f"time={obs['time']}."
            )

        if len(time_indices) > 1:
            raise ValueError(
                "Simulated production data contains "
                "duplicate time points."
            )

        time_index = time_indices[0]

        rows.append(
            data.values[time_index, :]
        )

    D = np.vstack(rows)

    return D