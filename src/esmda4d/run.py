from pathlib import Path

import numpy as np



def exclude_realizations(
    M,
    priors,
    realization_ids,
    failed_indices,
):
    """
    Remove failed realizations consistently from
    the model ensemble, full-grid priors, and
    persistent realization IDs.

    Parameters
    ----------
    M : ndarray
        Model ensemble with shape
        (n_model_parameters, n_ensemble).

    priors : dict
        Full-grid prior arrays. Each array must have
        ensemble dimension on the last axis.

    realization_ids : array-like
        Persistent ID corresponding to each ensemble
        column.

    failed_indices : array-like
        Current ensemble column indices to exclude.

    Returns
    -------
    M_surviving : ndarray
    priors_surviving : dict
    realization_ids_surviving : ndarray
    """
    M = np.asarray(
        M,
        dtype=float,
    )

    realization_ids = np.asarray(
        realization_ids,
    )

    failed_indices = np.asarray(
        failed_indices,
    )

    if M.ndim != 2:
        raise ValueError(
            "M must be a 2D array."
        )

    Ne = M.shape[1]

    if realization_ids.ndim != 1:
        raise ValueError(
            "realization_ids must be a 1D array."
        )

    if len(realization_ids) != Ne:
        raise ValueError(
            "realization_ids must contain one ID "
            "for each ensemble realization."
        )

    if failed_indices.ndim != 1:
        raise ValueError(
            "failed_indices must be a 1D array."
        )

    if not np.issubdtype(
        failed_indices.dtype,
        np.integer,
    ):
        raise TypeError(
            "failed_indices must contain integers."
        )

    failed_indices = failed_indices.astype(
        int,
        copy=False,
    )

    if len(np.unique(failed_indices)) != len(
        failed_indices
    ):
        raise ValueError(
            "failed_indices must be unique."
        )

    if np.any(failed_indices < 0):
        raise ValueError(
            "failed_indices cannot be negative."
        )

    if np.any(failed_indices >= Ne):
        raise ValueError(
            "failed_indices contains an index "
            "outside the ensemble."
        )

    keep = np.ones(
        Ne,
        dtype=bool,
    )

    keep[failed_indices] = False

    M_surviving = M[:, keep]

    realization_ids_surviving = (
        realization_ids[keep]
    )

    priors_surviving = {}

    for variable, prior in priors.items():

        prior = np.asarray(prior)

        if prior.ndim < 1:
            raise ValueError(
                f"Prior '{variable}' must have "
                "an ensemble dimension."
            )

        if prior.shape[-1] != Ne:
            raise ValueError(
                f"Prior '{variable}' ensemble "
                "size does not match M."
            )

        priors_surviving[variable] = (
            prior[..., keep]
        )

    return (
        M_surviving,
        priors_surviving,
        realization_ids_surviving,
    )


class AssimilationRun:
    """
    Manage the directory structure and persistent
    checkpoints of an ES-MDA run.
    """

    def __init__(
        self,
        path,
        n_assimilations,
    ):
        self.path = Path(path)

        if not isinstance(n_assimilations, int):
            raise TypeError(
                "n_assimilations must be an integer."
            )

        if n_assimilations < 1:
            raise ValueError(
                "n_assimilations must be at least 1."
            )

        self.n_assimilations = n_assimilations

    @property
    def state_path(self):
        return self.path / "esmda_state"

    @property
    def prior_path(self):
        return self.path / "ensemble_prior"

    @property
    def post_path(self):
        return self.path / "ensemble_post"

    def round_path(self, round_number):
        self._validate_round_number(
            round_number
        )

        return (
            self.path
            / f"ensemble_round_{round_number:03d}"
        )

    def checkpoint_path(self, round_number):
        self._validate_round_number(
            round_number
        )

        return (
            self.state_path
            / f"round_{round_number:03d}.npz"
        )

    @property
    def prior_checkpoint_path(self):
        return self.state_path / "prior.npz"

    @property
    def post_checkpoint_path(self):
        return self.state_path / "post.npz"

    @property
    def config_path(self):
        return self.state_path / "config.json"

    @property
    def observations_path(self):
        return self.state_path / "observations.npz"

    def _validate_round_number(
        self,
        round_number,
    ):
        if not isinstance(round_number, int):
            raise TypeError(
                "round_number must be an integer."
            )

        if (
            round_number < 1
            or round_number > self.n_assimilations
        ):
            raise ValueError(
                "round_number must be between 1 and "
                f"{self.n_assimilations}."
            )

    def create(self):
        """
        Create the complete directory structure.
        Existing directories are preserved.
        """
        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_path.mkdir(
            exist_ok=True,
        )

        self.prior_path.mkdir(
            exist_ok=True,
        )

        for round_number in range(
            1,
            self.n_assimilations + 1,
        ):
            self.round_path(
                round_number
            ).mkdir(
                exist_ok=True
            )

        self.post_path.mkdir(
            exist_ok=True,
        )

    def save_round_checkpoint(
        self,
        round_number,
        M,
        D,
        realization_ids,
    ):
        """
        Save a completed forward evaluation.

        The checkpoint represents:

            M_round -> forward model -> D_round

        realization_ids preserves the identity of each
        ensemble column, including after realizations
        have been permanently excluded.

        The final checkpoint file is created atomically.
        """
        self._validate_round_number(
            round_number
        )

        M = np.asarray(
            M,
            dtype=float,
        )

        D = np.asarray(
            D,
            dtype=float,
        )

        realization_ids = np.asarray(
            realization_ids,
            dtype=int,
        )

        if M.ndim != 2:
            raise ValueError(
                "M must be a 2D array."
            )

        if D.ndim != 2:
            raise ValueError(
                "D must be a 2D array."
            )

        if realization_ids.ndim != 1:
            raise ValueError(
                "realization_ids must be a 1D array."
            )

        if M.shape[1] != D.shape[1]:
            raise ValueError(
                "M and D must have the same "
                "ensemble size."
            )

        if len(realization_ids) != M.shape[1]:
            raise ValueError(
                "realization_ids must contain one ID "
                "for each ensemble realization."
            )

        if len(np.unique(realization_ids)) != len(
            realization_ids
        ):
            raise ValueError(
                "realization_ids must be unique."
            )

        if np.any(realization_ids < 1):
            raise ValueError(
                "realization_ids must be positive."
            )

        if not np.all(np.isfinite(M)):
            raise ValueError(
                "M must contain only finite values."
            )

        if not np.all(np.isfinite(D)):
            raise ValueError(
                "D must contain only finite values."
            )

        self.state_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = self.checkpoint_path(
            round_number
        )

        temporary = checkpoint.with_suffix(
            ".tmp.npz"
        )

        np.savez_compressed(
            temporary,
            M=M,
            D=D,
            realization_ids=realization_ids,
        )

        temporary.replace(
            checkpoint
        )


    def load_round_checkpoint(
        self,
        round_number,
    ):
        """
        Load a completed round checkpoint.

        Returns
        -------
        M
            Model ensemble.

        D
            Simulated-data ensemble.

        realization_ids
            Original realization ID corresponding to
            each column of M and D.
        """
        checkpoint = self.checkpoint_path(
            round_number
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{checkpoint}"
            )

        try:
            with np.load(
                checkpoint,
                allow_pickle=False,
            ) as data:

                required = {
                    "M",
                    "D",
                    "realization_ids",
                }

                if not required.issubset(
                    data.files
                ):
                    raise ValueError(
                        "Checkpoint does not contain "
                        "M, D, and realization_ids."
                    )

                M = np.asarray(
                    data["M"],
                    dtype=float,
                )

                D = np.asarray(
                    data["D"],
                    dtype=float,
                )

                realization_ids = np.asarray(
                    data["realization_ids"],
                    dtype=int,
                )

        except (OSError, EOFError) as error:
            raise ValueError(
                "Checkpoint could not be read: "
                f"{checkpoint}"
            ) from error

        if M.ndim != 2 or D.ndim != 2:
            raise ValueError(
                "Checkpoint M and D must be "
                "2D arrays."
            )

        if realization_ids.ndim != 1:
            raise ValueError(
                "Checkpoint realization_ids must "
                "be a 1D array."
            )

        if M.shape[1] != D.shape[1]:
            raise ValueError(
                "Checkpoint M and D have different "
                "ensemble sizes."
            )

        if len(realization_ids) != M.shape[1]:
            raise ValueError(
                "Checkpoint realization_ids do not "
                "match the ensemble size."
            )

        if len(np.unique(realization_ids)) != len(
            realization_ids
        ):
            raise ValueError(
                "Checkpoint realization_ids are "
                "not unique."
            )

        if np.any(realization_ids < 1):
            raise ValueError(
                "Checkpoint realization_ids must "
                "be positive."
            )

        if (
            not np.all(np.isfinite(M))
            or not np.all(np.isfinite(D))
        ):
            raise ValueError(
                "Checkpoint contains non-finite "
                "values."
            )

        return M, D, realization_ids   

    def completed_rounds(self):
        """
        Return sequential valid completed rounds.

        A later checkpoint is not considered completed
        if an earlier round is missing or invalid.
        """
        completed = []

        for round_number in range(
            1,
            self.n_assimilations + 1,
        ):
            try:
                self.load_round_checkpoint(
                    round_number
                )
            except (
                FileNotFoundError,
                ValueError,
            ):
                break

            completed.append(
                round_number
            )

        return completed

    def last_completed_round(self):
        """
        Return the last sequential valid completed
        round, or None if no round is complete.
        """
        completed = self.completed_rounds()

        if not completed:
            return None

        return completed[-1]