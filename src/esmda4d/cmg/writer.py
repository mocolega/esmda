from pathlib import Path
import re
import numpy as np

from esmda4d.model import unpack_model_ensemble
from esmda4d.grid import reconstruct_grid_property


class CMGModelWriter:
    """
    Create CMG input files for an ensemble of
    reservoir realizations.

    Each column of M corresponds to one realization.
    Persistent realization identity is provided by
    realization_ids.

    Example
    -------
    realization_ids = [1, 2, 4]

    M[:, 0] -> realization_0001
    M[:, 1] -> realization_0002
    M[:, 2] -> realization_0004
    """

    def __init__(
        self,
        template_path,
        output_dir,
        property_dir="props",
    ):
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.property_dir = property_dir

        self._validate()

    def _validate(self):

        if not self.template_path.is_file():
            raise FileNotFoundError(
                f"Template file not found: "
                f"{self.template_path}"
            )

    def write_ensemble(
        self,
        M,
        metadata,
        priors,
        realization_ids = None,
    ):
        """
        Write all realizations in model ensemble M.

        Parameters
        ----------
        M : ndarray
            ES-MDA model ensemble with shape
            (n_model_parameters, n_ensemble).

        metadata : list
            Metadata describing each row of M.

        priors : dict
            Full prior property ensembles.

            Each entry must have shape
            (NI, NJ, NK, n_ensemble).

            Example:
            {
                "porosity": prior_porosity,
                "permeability": prior_permeability,
                "ntg": prior_ntg,
            }
        realization_ids : list, optional
            List of realization IDs corresponding to the columns of M.
            If None, IDs will be assigned sequentially starting from 1.
        Returns
        -------
        realization_paths : list of Path
            Paths to the generated CMG .dat files.
        """

        properties = unpack_model_ensemble(
            M=M,
            metadata=metadata,
        )

        Ne = M.shape[1]

                # ----------------------------------------------
        # Realization identities
        # ----------------------------------------------

        if realization_ids is None:
            realization_ids = np.arange(
                1,
                Ne + 1,
                dtype=int,
            )
        else:
            realization_ids = np.asarray(
                realization_ids,
            )

            if realization_ids.ndim != 1:
                raise ValueError(
                    "realization_ids must be a "
                    "1D array."
                )

            if len(realization_ids) != Ne:
                raise ValueError(
                    "realization_ids must contain "
                    "one ID for each ensemble "
                    "realization."
                )

            if not np.issubdtype(
                realization_ids.dtype,
                np.integer,
            ):
                raise TypeError(
                    "realization_ids must contain "
                    "integers."
                )

            realization_ids = (
                realization_ids.astype(
                    int,
                    copy=False,
                )
            )

            if np.any(realization_ids < 1):
                raise ValueError(
                    "realization_ids must be "
                    "positive."
                )

            if (
                len(np.unique(realization_ids))
                != Ne
            ):
                raise ValueError(
                    "realization_ids must be "
                    "unique."
                )

        # ----------------------------------------------
        # Validate priors
        # ----------------------------------------------

        self._validate_priors(
            properties=properties,
            priors=priors,
            Ne=Ne,
        )

        # ----------------------------------------------
        # Reconstruct complete grid properties
        # ----------------------------------------------

        full_properties = {}

        for prop in properties:

            full_properties[prop.variable] = (
                reconstruct_grid_property(
                    grid_property=prop,
                    prior=priors[prop.variable],
                )
            )

        # ----------------------------------------------
        # Read and validate template
        # ----------------------------------------------

        template_text = (
            self.template_path.read_text()
        )

        self._validate_template(
            template_text=template_text,
            properties=properties,
        )

        # ----------------------------------------------
        # Create ensemble directory
        # ----------------------------------------------

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        realization_paths = []

        # ----------------------------------------------
        # Write each realization
        # ----------------------------------------------

        for j in range(Ne):

            realization_number = (
                realization_ids[j]
            )

            realization_id = (
                f"{realization_number:04d}"
            )

            realization_dir = (
                self.output_dir
                / f"realization_{realization_id}"
            )

            props_dir = (
                realization_dir
                / self.property_dir
            )

            props_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            replacements = {}

            for prop in properties:

                property_filename = (
                    f"{prop.variable}_"
                    f"{realization_id}.inc"
                )

                property_path = (
                    props_dir
                    / property_filename
                )

                # Complete NI x NJ x NK property for
                # this realization, flattened according
                # to CMG ordering:
                #
                # I varies fastest, then J, then K.

                values = (
                    full_properties[
                        prop.variable
                    ][:, :, :, j]
                    .ravel(order="F")
                )

                self._write_property(
                    path=property_path,
                    values=values,
                )

                include_path = (
                    Path(self.property_dir)
                    / property_filename
                )

                include_path = (
                    include_path.as_posix()
                )

                placeholder = (
                    f"$${prop.variable}"
                )

                replacements[
                    placeholder
                ] = include_path

            # ------------------------------------------
            # Create realization .dat file
            # ------------------------------------------

            model_text = template_text

            for placeholder, replacement in (
                replacements.items()
            ):
                model_text = model_text.replace(
                    placeholder,
                    replacement,
                )

            unresolved = re.findall(
                r"\$\$[A-Za-z_][A-Za-z0-9_]*",
                model_text,
            )

            if unresolved:
                raise ValueError(
                    "Generated CMG model contains "
                    "unresolved template placeholders: "
                    f"{', '.join(sorted(set(unresolved)))}."
                )

            model_filename = (
                f"model_{realization_id}.dat"
            )

            model_path = (
                realization_dir
                / model_filename
            )

            model_path.write_text(
                model_text
            )

            realization_paths.append(
                model_path
            )

        return realization_paths

    @staticmethod
    def _write_property(
        path,
        values,
        tokens_per_line=20,
    ):
        """
        Write one grid-property include file.

        Consecutive repeated values are compressed
        using CMG N*V notation.
        """

        tokens = []

        start = 0

        while start < len(values):

            value = values[start]
            end = start + 1

            while (
                end < len(values)
                and values[end] == value
            ):
                end += 1

            count = end - start

            value_text = f"{value:.8g}"

            if count > 1:
                token = f"{count}*{value_text}"
            else:
                token = value_text

            tokens.append(token)

            start = end

        with path.open("w") as file:

            for i in range(
                0,
                len(tokens),
                tokens_per_line,
            ):
                line = tokens[
                    i:i + tokens_per_line
                ]

                file.write(
                    " ".join(line) + "\n"
                )

    @staticmethod
    def _validate_template(
        template_text,
        properties,
    ):
        """
        Validate that template placeholders and model
        properties match exactly.
        """

        placeholders = set(
            re.findall(
                r"\$\$[A-Za-z_][A-Za-z0-9_]*",
                template_text,
            )
        )

        property_placeholders = {
            f"$${prop.variable}"
            for prop in properties
        }

        missing_properties = (
            placeholders
            - property_placeholders
        )

        if missing_properties:
            missing = ", ".join(
                sorted(missing_properties)
            )

            raise ValueError(
                "Template contains placeholders with "
                "no corresponding model property: "
                f"{missing}."
            )

        missing_placeholders = (
            property_placeholders
            - placeholders
        )

        if missing_placeholders:
            missing = ", ".join(
                sorted(missing_placeholders)
            )

            raise ValueError(
                "Model properties have no corresponding "
                "template placeholder: "
                f"{missing}."
            )

    @staticmethod
    def _validate_priors(
        properties,
        priors,
        Ne,
    ):
        """
        Validate full prior property ensembles.
        """

        if not isinstance(priors, dict):
            raise TypeError(
                "priors must be a dictionary."
            )

        grid_shape = None

        for prop in properties:

            if prop.variable not in priors:
                raise ValueError(
                    "Missing prior property for "
                    f"{prop.variable}."
                )

            prior = np.asarray(
                priors[prop.variable],
                dtype=float,
            )

            if prior.ndim != 4:
                raise ValueError(
                    f"Prior for {prop.variable} must "
                    "have shape "
                    "(NI, NJ, NK, n_ensemble)."
                )

            if prior.shape[3] != Ne:
                raise ValueError(
                    f"Prior for {prop.variable} must "
                    "contain the same number of "
                    "ensemble realizations as M."
                )

            if grid_shape is None:
                grid_shape = prior.shape[:3]

            elif prior.shape[:3] != grid_shape:
                raise ValueError(
                    "All prior properties must use "
                    "the same reservoir grid shape."
                )