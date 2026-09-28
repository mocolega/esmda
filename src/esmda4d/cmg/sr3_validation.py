from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .sr3 import SR3Reader

@dataclass
class SR3ProductionRequirement:
    origin: str
    entity: str
    variable: str
    dates: list

@dataclass
class SR3SpatialRequirement:
    variable: str
    dates: list

@dataclass
class SR3Validation:
    valid: bool
    sr3_path: Path
    reason: str | None = None

def _normalize_date(date):
    try:
        return np.datetime64(
            date,
            "ns",
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid date: {date}"
        )

class SR3Validator:
    def validate(
        self,
        model_path,
        production_requirements=None,
        spatial_requirements=None,
    ):
        """
        Validate the basic SR3 output associated
        with a CMG model file.

        This initial validation checks only that:

        1. The SR3 file exists.
        2. The SR3 file is readable.
        3. Its MasterTimeTable can be read.

        Production and spatial-data requirements
        are validated separately in later steps.
        """

        model_path = Path(model_path)

        sr3_path = model_path.with_suffix(
            ".sr3"
        )

        if not sr3_path.exists():
            return SR3Validation(
                valid=False,
                sr3_path=sr3_path,
                reason="SR3 file not found.",
            )

        try:
            reader = SR3Reader(
                sr3_path
            )

            reader.master_timetable()

            if production_requirements is not None:
                for requirement in (
                    production_requirements
                ):
                    try:
                        dates, values = (
                            reader.read_time_series(
                                origin=requirement.origin,
                                variable=requirement.variable,
                                entity=requirement.entity,
                            )
                        )
                    except Exception as error:
                        return SR3Validation(
                            valid=False,
                            sr3_path=sr3_path,
                            reason=(
                                "Required production data "
                                "could not be read: "
                                f"{error}"
                            ),
                        )

                    normalized_dates = [
                        _normalize_date(date)
                        for date in dates
                    ]

                    for required_date in requirement.dates:
                        normalized_required_date = (
                            _normalize_date(
                                required_date
                            )
                        )

                        try:
                            date_index = (
                                normalized_dates.index(
                                    normalized_required_date
                                )
                            )
                        except ValueError:
                            return SR3Validation(
                                valid=False,
                                sr3_path=sr3_path,
                                reason=(
                                    "Required production date "
                                    f"{required_date} not found."
                                ),
                            )

                        if not np.isfinite(
                            values[date_index]
                        ):
                            return SR3Validation(
                                valid=False,
                                sr3_path=sr3_path,
                                reason=(
                                    "Required production value "
                                    "is not finite for date "
                                    f"{required_date}."
                                ),
                            )

            if spatial_requirements is not None:
                for requirement in (
                    spatial_requirements
                ):
                    for required_date in (
                        requirement.dates
                    ):
                        try:
                            values = (
                                reader.read_spatial_property(
                                    variable=requirement.variable,
                                    date=required_date,
                                )
                            )
                        except Exception as error:
                            return SR3Validation(
                                valid=False,
                                sr3_path=sr3_path,
                                reason=(
                                    "Required spatial data "
                                    "could not be read: "
                                    f"{error}"
                                ),
                            )

                        finite_values = values[
                            np.isfinite(values)
                        ]

                        if finite_values.size == 0:
                            return SR3Validation(
                                valid=False,
                                sr3_path=sr3_path,
                                reason=(
                                    "Required spatial property "
                                    "contains no finite values for "
                                    f"{requirement.variable}, "
                                    f"{required_date}."
                                ),
                            )
        
        except Exception as error:
            return SR3Validation(
                valid=False,
                sr3_path=sr3_path,
                reason=(
                    "SR3 file could not be read: "
                    f"{error}"
                ),
            )



        return SR3Validation(
            valid=True,
            sr3_path=sr3_path,
        )