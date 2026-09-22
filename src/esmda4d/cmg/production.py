from pathlib import Path

import numpy as np

from esmda4d.production import SimulatedProductionData
from esmda4d.cmg.sr3 import SR3Reader


class CMGProductionReader:
    """
    Convert production results from multiple CMG SR3
    files into generic SimulatedProductionData objects.

    The CMG-specific mapping between generic entity
    types and SR3 TimeSeries origins is kept here.
    """

    DEFAULT_ORIGIN_MAP = {
        "well": "WELLS",
        "sector": "SECTORS",
    }

    def __init__(
        self,
        origin_map=None,
    ):
        if origin_map is None:
            origin_map = self.DEFAULT_ORIGIN_MAP

        self.origin_map = dict(origin_map)

    def _get_origin(self, entity_type):
        """
        Translate a generic entity type into the
        corresponding CMG SR3 TimeSeries origin.
        """
        if entity_type not in self.origin_map:
            raise ValueError(
                "No CMG SR3 origin mapping for "
                f"entity type {entity_type}."
            )

        return self.origin_map[entity_type]

    def read_ensemble(
        self,
        sr3_paths,
        metadata,
    ):
        """
        Read the production quantities required by
        observation metadata from an ensemble of
        CMG SR3 files.

        Parameters
        ----------
        sr3_paths
            One SR3 path for each ensemble realization.
            The path order defines the ensemble-column
            order.

        metadata
            Metadata produced by
            build_production_observations().

        Returns
        -------
        list[SimulatedProductionData]
            One object for each unique combination of
            entity, entity_type, and variable.
        """
        sr3_paths = [
            Path(path)
            for path in sr3_paths
        ]

        if len(sr3_paths) < 2:
            raise ValueError(
                "At least two SR3 files are required "
                "to build a simulated ensemble."
            )

        if len(metadata) == 0:
            raise ValueError(
                "Production metadata cannot be empty."
            )

        # -------------------------------------------------
        # Determine which simulated datasets are needed.
        #
        # Several metadata rows may refer to different
        # dates of the same well/variable combination.
        # We only need to read that SR3 time series once
        # per realization.
        # -------------------------------------------------

        requested_datasets = []

        for item in metadata:
            key = (
                item["entity"],
                item["entity_type"],
                item["variable"],
            )

            if key not in requested_datasets:
                requested_datasets.append(key)

        simulated_data = []

        # -------------------------------------------------
        # Read each requested dataset across all
        # realizations.
        # -------------------------------------------------

        for (
            entity,
            entity_type,
            variable,
        ) in requested_datasets:

            origin = self._get_origin(
                entity_type
            )

            ensemble_values = []
            reference_dates = None

            for sr3_path in sr3_paths:
                reader = SR3Reader(
                    sr3_path
                )

                dates, values = (
                    reader.read_time_series(
                        origin=origin,
                        variable=variable,
                        entity=entity,
                    )
                )

                if reference_dates is None:
                    reference_dates = dates

                elif dates != reference_dates:
                    raise ValueError(
                        "SR3 ensemble realizations "
                        "do not have matching "
                        "production dates for "
                        f"{entity} / {variable}."
                    )

                ensemble_values.append(
                    values
                )

            # Each list entry is one realization:
            #
            # [
            #   realization 1 -> (n_times,),
            #   realization 2 -> (n_times,),
            #   ...
            # ]
            #
            # column_stack converts this into:
            #
            # (n_times, n_ensemble)

            values = np.column_stack(
                ensemble_values
            )

            simulated_data.append(
                SimulatedProductionData(
                    entity=entity,
                    entity_type=entity_type,
                    variable=variable,
                    time=np.asarray(
                        reference_dates,
                        dtype="datetime64[ns]",
                    ),
                    values=values,
                )
            )

        return simulated_data