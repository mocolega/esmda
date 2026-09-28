from pathlib import Path

from esmda4d.forward import ForwardModel
from esmda4d.production import (
    build_production_ensemble,
)
from esmda4d.cmg.production import (
    CMGProductionReader,
)

from .sr3_validation import (
    SR3ProductionRequirement,
    SR3Validator,
)


class CMGRealizationFailure(RuntimeError):
    """
    Raised when one or more CMG realizations
    remain failed after all retry attempts.
    """

    def __init__(
        self,
        failed_indices,
        failed_ids,
    ):
        self.failed_indices = list(
            failed_indices
        )

        self.failed_ids = list(
            failed_ids
        )

        super().__init__(
            "CMG realization(s) failed after "
            "all retry attempts: "
            f"{self.failed_ids}."
        )


class CMGForwardModel(ForwardModel):
    """
    CMG forward model for production-data assimilation.

    Converts an ES-MDA model ensemble M into the
    simulated-data ensemble D.
    """

    def __init__(
        self,
        writer,
        runner,
        model_metadata,
        production_metadata,
        priors,
        realization_ids=None,
        max_retries=0,
    ):
        self.writer = writer
        self.runner = runner

        self.model_metadata = model_metadata
        self.production_metadata = (
            production_metadata
        )

        self.priors = priors
        self.realization_ids = realization_ids

        
        self.production_reader = (
           CMGProductionReader()
        )   
        self.sr3_validator = SR3Validator()

        if not isinstance(max_retries, int):
            raise TypeError(
                "max_retries must be an integer."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        self.max_retries = max_retries


    def _production_requirements(self):
        requirements = {}

        for item in self.production_metadata:
            entity = item["entity"]
            entity_type = item["entity_type"]
            variable = item["variable"]
            date = item["time"]

            try:
                origin = (
                    self.production_reader
                    ._get_origin(entity_type)
                )
            except KeyError:
                raise ValueError(
                    "Unsupported CMG production "
                    f"entity type: {entity_type}"
                )

            key = (
                origin,
                entity,
                variable,
            )

            if key not in requirements:
                requirements[key] = (
                    SR3ProductionRequirement(
                        origin=origin,
                        entity=entity,
                        variable=variable,
                        dates=[],
                    )
                )

            requirements[key].dates.append(
                date
            )

        return list(
            requirements.values()
        )

    def _validate_sr3_results(
    self,
    model_paths,
    results,
    ):
        production_requirements = (
            self._production_requirements()
        )

        valid = []

        for model_path, result in zip(
            model_paths,
            results,
        ):
            if not result.succeeded:
                valid.append(False)
                continue

            validation = (
                self.sr3_validator.validate(
                    model_path,
                    production_requirements=(
                        production_requirements
                    ),
                )
            )

            valid.append(
                validation.valid
            )

        return valid
    
    def set_ensemble_context(
        self,
        priors,
        realization_ids,
    ):
        """
        Update ensemble-dependent information after
        realizations have been excluded.

        Parameters
        ----------
        priors : dict
            Full-grid prior arrays. The ensemble
            dimension must be the last dimension.

        realization_ids : array-like
            Persistent realization IDs corresponding
            to the current ensemble columns.
        """
        realization_ids = list(
            realization_ids
        )

        if not realization_ids:
            raise ValueError(
                "realization_ids cannot be empty."
            )

        if len(set(realization_ids)) != len(
            realization_ids
        ):
            raise ValueError(
                "realization_ids must be unique."
            )

        if any(
            realization_id < 1
            for realization_id in realization_ids
        ):
            raise ValueError(
                "realization_ids must be positive."
            )

        for variable, prior in priors.items():
            if prior.shape[-1] != len(
                realization_ids
            ):
                raise ValueError(
                    f"Prior '{variable}' ensemble "
                    "size does not match "
                    "realization_ids."
                )

        self.priors = priors
        self.realization_ids = realization_ids

    def _run_with_retries(
        self,
        model_paths,
    ):
        """
        Run all models once, then retry only
        failed realizations.

        Returns results in the same order as
        model_paths.
        """

        model_paths = list(model_paths)

        results = self.runner.run_ensemble(
            model_paths
        )

        for _ in range(
            self.max_retries
        ):
            valid = (
                self._validate_sr3_results(
                    model_paths,
                    results,
                )
            )

            failed_indices = [
                j
                for j, is_valid
                in enumerate(valid)
                if not is_valid
            ]

            if not failed_indices:
                break

            retry_paths = [
                model_paths[j]
                for j in failed_indices
            ]

            retry_results = (
                self.runner.run_ensemble(
                    retry_paths
                )
            )

            for j, retry_result in zip(
                failed_indices,
                retry_results,
            ):
                results[j] = retry_result

        return results

    def _check_results(
        self,
        model_paths,
        results,
        realization_ids,
    ):
        valid = (
            self._validate_sr3_results(
                model_paths,
                results,
            )
        )

        failed_indices = [
            j
            for j, is_valid
            in enumerate(valid)
            if not is_valid
        ]

        if failed_indices:
            failed_ids = [
                realization_ids[j]
                for j in failed_indices
            ]

            raise CMGRealizationFailure(
                failed_indices=failed_indices,
                failed_ids=failed_ids,
            )

    def run(self, M):
        Ne = M.shape[1]

        if self.realization_ids is None:
            realization_ids = list(
                range(1, Ne + 1)
            )
        else:
            realization_ids = list(
                self.realization_ids
            )

            if len(realization_ids) != Ne:
                raise ValueError(
                    "realization_ids must contain "
                    "one ID for each ensemble "
                    "realization."
                )

        # -----------------------------------------
        # 1. Write CMG input models
        # -----------------------------------------

        model_paths = (
            self.writer.write_ensemble(
                M=M,
                metadata=self.model_metadata,
                priors=self.priors,
                realization_ids=realization_ids,
            )
        )

        # -----------------------------------------
        # 2. Run CMG
        # -----------------------------------------
        results = (
            self._run_with_retries(
                model_paths
            )
        )

        self._check_results(
            model_paths=model_paths,
            results=results,
            realization_ids=realization_ids,
        )

        # -----------------------------------------
        # 3. Check for failed realizations
        # -----------------------------------------

        failed_indices = [
            j
            for j, result in enumerate(results)
            if not result.succeeded
        ]

        if failed_indices:
            failed_ids = [
                realization_ids[j]
                for j in failed_indices
            ]

            raise CMGRealizationFailure(
                failed_indices=failed_indices,
                failed_ids=failed_ids,
            )

        # -----------------------------------------
        # 4. Locate SR3 output files
        # -----------------------------------------

        sr3_paths = [
            Path(model_path).with_suffix(
                ".sr3"
            )
            for model_path in model_paths
        ]

        missing = [
            path
            for path in sr3_paths
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "CMG did not produce the expected "
                f"SR3 file: {missing[0]}"
            )

        # -----------------------------------------
        # 5. Convert CMG production results into
        #    generic simulated production data
        # -----------------------------------------

        production_reader = (
            CMGProductionReader()
        )

        simulated = (
            production_reader.read_ensemble(
                sr3_paths=sr3_paths,
                metadata=self.production_metadata,
            )
        )

        # -----------------------------------------
        # 6. Construct D
        # -----------------------------------------

        return build_production_ensemble(
            simulated,
            self.production_metadata,
        )