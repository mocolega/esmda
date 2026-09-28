import numpy as np

from .sr3 import SR3Reader


class CMGStateReader:
    """
    Read simulator states required by a data
    processor from a CMG SR3 file.
    """

    def read(
        self,
        model_path,
        requirements,
    ):
        sr3_path = model_path.with_suffix(
            ".sr3"
        )

        reader = SR3Reader(
            sr3_path
        )

        states = {}

        for requirement in requirements:
            variable = requirement.variable

            states[variable] = {}

            for date in requirement.dates:
                reader_date = (
                    self._find_reader_date(
                        reader,
                        date,
                    )
                )

                values = (
                    reader.read_spatial_property(
                        variable=variable,
                        date=reader_date,
                    )
                )

                states[variable][date] = (
                    values
                )

        return states

    @staticmethod
    def _find_reader_date(
        reader,
        requested_date,
    ):
        requested_date = np.datetime64(
            requested_date,
            "ns",
        )

        for reader_date in (
            reader.spatial_dates()
        ):
            candidate = np.datetime64(
                reader_date,
                "ns",
            )

            if candidate == requested_date:
                return reader_date

        raise ValueError(
            "Required spatial date "
            f"{requested_date} not found."
        )