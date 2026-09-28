from pathlib import Path
import datetime as dt

import h5py
import numpy as np


class SR3Reader:
    """
    Low-level reader for CMG SR3 files.

    This class provides access to the raw time-series
    information stored in an SR3 file. It does not
    perform production-data processing or ES-MDA
    observation matching.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._validate()

    def _validate(self):
        if not self.path.is_file():
            raise FileNotFoundError(
                f"SR3 file not found: {self.path}"
            )

    @staticmethod
    def _decode_strings(values):
        """
        Convert an SR3 byte-string array into
        regular Python strings.
        """
        return [
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ]

    @staticmethod
    def _convert_date(value):
        """
        Convert CMG numeric date format into
        a Python datetime.
        """
        year = int(value / 10000)
        value = value - year * 10000

        month = int(value / 100)
        value = value - month * 100

        day = int(value)
        value = (value - day) * 24

        hour = int(value)
        value = (value - hour) * 60

        minute = int(value)
        value = (value - minute) * 60

        second = int(value)

        return dt.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
        )

    def time_series_origins(self):
        """
        Return the available groups under /TimeSeries.
        """
        with h5py.File(self.path, "r") as file:
            if "TimeSeries" not in file:
                return []

            return list(
                file["TimeSeries"].keys()
            )

    def variables(self, origin):
        """
        Return the variables available for a
        TimeSeries origin, for example WELLS.
        """
        with h5py.File(self.path, "r") as file:
            group_path = f"TimeSeries/{origin}"

            if group_path not in file:
                raise ValueError(
                    f"Time-series origin not found: "
                    f"{origin}"
                )

            group = file[group_path]

            if "Variables" not in group:
                raise ValueError(
                    f"Origin {origin} does not contain "
                    "a Variables dataset."
                )

            return self._decode_strings(
                group["Variables"][:]
            )

    def entities(self, origin):
        """
        Return the entities stored for a TimeSeries
        origin.

        For origin='WELLS', these are well names.
        """
        with h5py.File(self.path, "r") as file:
            group_path = f"TimeSeries/{origin}"

            if group_path not in file:
                raise ValueError(
                    f"Time-series origin not found: "
                    f"{origin}"
                )

            group = file[group_path]

            if "Origins" not in group:
                raise ValueError(
                    f"Origin {origin} does not contain "
                    "an Origins dataset."
                )

            return self._decode_strings(
                group["Origins"][:]
            )

    def master_timetable(self):
        """
        Return a dictionary mapping CMG timestep
        indices to Python datetime objects.
        """
        with h5py.File(self.path, "r") as file:
            table = file[
                "General/MasterTimeTable"
            ]

            indices = table["Index"]
            dates = table["Date"]

            return {
                int(index): self._convert_date(date)
                for index, date
                in zip(indices, dates)
            }

    def dates(self, origin):
        """
        Return the dates stored for a TimeSeries
        origin.
        """
        timetable = self.master_timetable()

        with h5py.File(self.path, "r") as file:
            group_path = f"TimeSeries/{origin}"

            if group_path not in file:
                raise ValueError(
                    f"Time-series origin not found: "
                    f"{origin}"
                )

            timesteps = file[
                f"{group_path}/Timesteps"
            ][:]

        return [
            timetable[int(timestep)]
            for timestep in timesteps
        ]
    
    def spatial_dates(self):
        """
        Return dates for which spatial grid data
        are stored in the SR3 file.
        """
        timetable = self.master_timetable()

        dates = []

        with h5py.File(self.path, "r") as file:
            if "SpatialProperties" not in file:
                return dates

            spatial = file["SpatialProperties"]

            for key in spatial.keys():
                try:
                    timestep = int(key)
                except ValueError:
                    continue

                if timestep not in timetable:
                    continue

                dates.append(
                    timetable[timestep]
                )

        return dates
    
    def read_time_series(
        self,
        origin,
        variable,
        entity,
    ):
        """
        Read one time series from the SR3 file.

        Parameters
        ----------
        origin
            CMG TimeSeries origin, such as "WELLS".

        variable
            Variable stored under that origin,
            such as "BHP" or "OILRATSC".

        entity
            Entity name, such as a well name.

        Returns
        -------
        dates : list[datetime.datetime]
            Dates associated with the time series.

        values : numpy.ndarray
            One-dimensional array containing the
            simulated values.
        """
        with h5py.File(self.path, "r") as file:
            group_path = f"TimeSeries/{origin}"

            if group_path not in file:
                raise ValueError(
                    f"Time-series origin not found: "
                    f"{origin}"
                )

            group = file[group_path]

            variables = self._decode_strings(
                group["Variables"][:]
            )

            entities = self._decode_strings(
                group["Origins"][:]
            )

            if variable not in variables:
                raise ValueError(
                    f"Variable {variable} not found "
                    f"in origin {origin}."
                )

            if entity not in entities:
                raise ValueError(
                    f"Entity {entity} not found "
                    f"in origin {origin}."
                )

            variable_index = variables.index(
                variable
            )

            entity_index = entities.index(
                entity
            )

            values = np.asarray(
                group["Data"][
                    :,
                    variable_index,
                    entity_index,
                ],
                dtype=float,
            )

        dates = self.dates(origin)

        return dates, values

    def spatial_variables(self, date):
        """
        Return spatial variables stored at a
        requested date.
        """
        timetable = self.master_timetable()

        timestep = None

        for index, timestep_date in (
            timetable.items()
        ):
            if timestep_date == date:
                timestep = index
                break

        if timestep is None:
            raise ValueError(
                f"Date {date} not found in "
                "MasterTimeTable."
            )

        timestep_key = f"{timestep:06d}"

        with h5py.File(self.path, "r") as file:
            if "SpatialProperties" not in file:
                raise ValueError(
                    "SR3 file does not contain "
                    "SpatialProperties."
                )

            spatial = file["SpatialProperties"]

            if timestep_key not in spatial:
                raise ValueError(
                    f"No spatial data stored "
                    f"for date {date}."
                )

            group = spatial[timestep_key]

            return [
                key
                for key in group.keys()
                if key != "GRID"
            ]

    def read_spatial_property(
    self,
    variable,
    date,
    grid="matrix",
    null_value=np.nan,
    ):
        """
        Read one spatial grid property at a
        requested date.

        Returns an array with shape:

            (NI, NJ, NK)

        For the matrix grid, inactive cells are
        filled with null_value.
        """

        if grid != "matrix":
            raise ValueError(
                "Only grid='matrix' is currently "
                "supported."
            )

        timetable = self.master_timetable()

        timestep = None

        for index, timestep_date in (
            timetable.items()
        ):
            if timestep_date == date:
                timestep = index
                break

        if timestep is None:
            raise ValueError(
                f"Date {date} not found in "
                "MasterTimeTable."
            )

        timestep_key = f"{timestep:06d}"

        with h5py.File(self.path, "r") as file:

            if "SpatialProperties" not in file:
                raise ValueError(
                    "SR3 file does not contain "
                    "SpatialProperties."
                )

            spatial = file["SpatialProperties"]

            if timestep_key not in spatial:
                raise ValueError(
                    f"No spatial data stored "
                    f"for date {date}."
                )

            state = spatial[timestep_key]

            if variable not in state:
                raise ValueError(
                    f"Spatial variable "
                    f"'{variable}' not found "
                    f"for date {date}."
                )

            if "GRID" not in state:
                raise ValueError(
                    "Spatial state does not "
                    "contain GRID information."
                )

            grid_data = state["GRID"]

            ni = int(grid_data["IGNTID"][0])
            nj = int(grid_data["IGNTJD"][0])
            nk = int(grid_data["IGNTKD"][0])

            cell_ids = np.asarray(
                grid_data["IPSTCS"][:],
                dtype=int,
            )

            values = np.asarray(
                state[variable][:],
                dtype=float,
            )

            if len(values) != len(cell_ids):
                raise ValueError(
                    "Spatial property values and IPSTCS "
                    "have inconsistent sizes."
                )

            n_cells = ni * nj * nk

            # Matrix-grid cells use CMG cell IDs
            # from 1 to NI*NJ*NK.

            matrix_mask = (
                cell_ids <= n_cells
            )

            matrix_cell_ids = (
                cell_ids[matrix_mask]
            )

            matrix_values = (
                values[matrix_mask]
            )

            if not np.all(
                np.isfinite(matrix_values)
            ):
                raise ValueError(
                    "Spatial property contains "
                    "non-finite values in active "
                    "matrix cells."
                )

            full_flat = np.full(
                n_cells,
                null_value,
                dtype=float,
            )

            # CMG cell IDs are 1-based.
            full_flat[
                matrix_cell_ids - 1
            ] = matrix_values

            # Internal library convention:
            #
            # shape = (NI, NJ, NK)
            # I varies fastest.

            return full_flat.reshape(
                (ni, nj, nk),
                order="F",
            )