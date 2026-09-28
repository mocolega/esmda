from dataclasses import dataclass

import numpy as np
import json
from pathlib import Path

@dataclass
class DerivedObservation:
    values: np.ndarray
    standard_deviations: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(
            self.values,
            dtype=float,
        )

        self.standard_deviations = np.asarray(
            self.standard_deviations,
            dtype=float,
        )

        if self.values.ndim != 1:
            raise ValueError(
                "Derived observation values must "
                "be one-dimensional."
            )

        if self.standard_deviations.ndim != 1:
            raise ValueError(
                "Derived observation standard "
                "deviations must be "
                "one-dimensional."
            )

        if (
            self.values.shape
            != self.standard_deviations.shape
        ):
            raise ValueError(
                "Derived observation values and "
                "standard deviations must have "
                "the same shape."
            )

        if not np.all(
            np.isfinite(self.values)
        ):
            raise ValueError(
                "Derived observation values must "
                "be finite."
            )

        if not np.all(
            np.isfinite(
                self.standard_deviations
            )
        ):
            raise ValueError(
                "Derived observation standard "
                "deviations must be finite."
            )

        if np.any(
            self.standard_deviations <= 0.0
        ):
            raise ValueError(
                "Derived observation standard "
                "deviations must be positive."
            )

def build_observations(
    production_d_obs,
    production_Ce,
    derived_observation=None,
):
    production_d_obs = np.asarray(
        production_d_obs,
        dtype=float,
    )

    production_Ce = np.asarray(
        production_Ce,
        dtype=float,
    )

    if production_d_obs.ndim != 1:
        raise ValueError(
            "Production observation vector must "
            "be one-dimensional."
        )

    n_production = len(
        production_d_obs
    )

    if production_Ce.shape != (
        n_production,
        n_production,
    ):
        raise ValueError(
            "Production covariance matrix has "
            "an incompatible shape."
        )

    if derived_observation is None:
        return (
            production_d_obs.copy(),
            production_Ce.copy(),
        )

    if not isinstance(
        derived_observation,
        DerivedObservation,
    ):
        raise TypeError(
            "derived_observation must be a "
            "DerivedObservation object."
        )

    d_obs = np.concatenate(
        (
            production_d_obs,
            derived_observation.values,
        )
    )

    derived_variances = (
        derived_observation.standard_deviations
        ** 2
    )

    n_derived = len(
        derived_observation.values
    )

    Ce = np.zeros(
        (
            n_production + n_derived,
            n_production + n_derived,
        ),
        dtype=float,
    )

    Ce[
        :n_production,
        :n_production,
    ] = production_Ce

    Ce[
        n_production:,
        n_production:,
    ] = np.diag(
        derived_variances
    )

    return d_obs, Ce

def read_derived_observation(path):
    """
    Read derived observations from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON observation file.

    Returns
    -------
    DerivedObservation
        Derived observation values and standard
        deviations.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "derived_data" not in data:
        raise ValueError(
            "Observation file must contain "
            "'derived_data'."
        )

    derived_data = data["derived_data"]

    required_fields = (
        "values",
        "standard_deviations",
    )

    for field in required_fields:
        if field not in derived_data:
            raise ValueError(
                "Derived observation entry is "
                f"missing required field "
                f"'{field}'."
            )

    return DerivedObservation(
        values=np.asarray(
            derived_data["values"],
            dtype=float,
        ),
        standard_deviations=np.asarray(
            derived_data[
                "standard_deviations"
            ],
            dtype=float,
        ),
    )