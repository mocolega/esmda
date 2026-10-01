from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .esmda import esmda_update
from .inflation import validate_alphas


@dataclass
class EnsembleEvaluation:
    """
    Result of a successful ensemble forward
    evaluation.
    """

    M: np.ndarray
    D: np.ndarray
    priors: dict
    realization_ids: np.ndarray
    excluded_ids: list


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

def evaluate_ensemble(
    M,
    priors,
    realization_ids,
    forward_model,
    failure_exception,
):
    """
    Evaluate an ensemble, permanently excluding
    realizations reported as failed.

    The forward model is retried with the surviving
    ensemble until a complete simulated-data
    ensemble is obtained.
    """
    M_current = np.asarray(
        M,
        dtype=float,
    )

    priors_current = priors

    realization_ids_current = np.asarray(
        realization_ids,
    )

    excluded_ids = []

    while True:
        forward_model.set_ensemble_context(
            priors=priors_current,
            realization_ids=realization_ids_current,
        )

        try:
            D = forward_model(
                M_current
            )

        except failure_exception as error:
            excluded_ids.extend(
                error.failed_ids
            )

            (
                M_current,
                priors_current,
                realization_ids_current,
            ) = exclude_realizations(
                M=M_current,
                priors=priors_current,
                realization_ids=(
                    realization_ids_current
                ),
                failed_indices=(
                    error.failed_indices
                ),
            )

            if M_current.shape[1] < 2:
                raise RuntimeError(
                    "Fewer than two realizations "
                    "remain after exclusions."
                ) from error

            continue

        D = np.asarray(
            D,
            dtype=float,
        )

        if D.ndim != 2:
            raise ValueError(
                "Forward model must return "
                "a 2D array."
            )

        if D.shape[1] != M_current.shape[1]:
            raise ValueError(
                "M and D must have the same "
                "ensemble size after forward "
                "evaluation."
            )

        return EnsembleEvaluation(
            M=M_current,
            D=D,
            priors=priors_current,
            realization_ids=(
                realization_ids_current
            ),
            excluded_ids=excluded_ids,
        )

def assimilate_round(
    M,
    priors,
    realization_ids,
    forward_model,
    failure_exception,
    d_obs,
    Ce,
    alpha,
    rng,
    inversion="direct",
    energy=0.99,
):
    """
    Evaluate the ensemble and perform one
    ES-MDA update.

    Returns
    -------
    evaluation : EnsembleEvaluation
        Successful forward evaluation before
        the update.

    M_updated : ndarray
        Updated model ensemble.
    """

    evaluation = evaluate_ensemble(
        M=M,
        priors=priors,
        realization_ids=realization_ids,
        forward_model=forward_model,
        failure_exception=failure_exception,
    )

    M_updated, _ = esmda_update(
        M=evaluation.M,
        D=evaluation.D,
        d_obs=d_obs,
        Ce=Ce,
        alpha=alpha,
        rng=rng,
        inversion=inversion,
        energy=energy,
    )

    return evaluation, M_updated


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
        return self.path / "prior"

    @property
    def post_path(self):
        return self.path / "post"

    def round_path(self, round_number):
        self._validate_round_number(
            round_number
        )

        return (
            self.path
            / f"round_{round_number:03d}"
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

    def save_observations(
        self,
        d_obs,
        Ce,
    ):
        """
        Save the observations used by ES-MDA.

        Parameters
        ----------
        d_obs : ndarray
            Observed-data vector with shape
            (n_data,).

        Ce : ndarray
            Observation-error covariance matrix with
            shape (n_data, n_data).
        """

        d_obs = np.asarray(
            d_obs,
            dtype=float,
        )

        Ce = np.asarray(
            Ce,
            dtype=float,
        )

        if d_obs.ndim != 1:
            raise ValueError(
                "d_obs must be a 1D array."
            )

        n_data = len(d_obs)

        if Ce.shape != (
            n_data,
            n_data,
        ):
            raise ValueError(
                "Ce must have shape "
                "(n_data, n_data)."
            )

        if not np.all(
            np.isfinite(d_obs)
        ):
            raise ValueError(
                "d_obs must contain only "
                "finite values."
            )

        if not np.all(
            np.isfinite(Ce)
        ):
            raise ValueError(
                "Ce must contain only "
                "finite values."
            )

        if not np.allclose(
            Ce,
            Ce.T,
        ):
            raise ValueError(
                "Ce must be symmetric."
            )

        if np.any(
            np.diag(Ce) <= 0.0
        ):
            raise ValueError(
                "Ce diagonal entries must "
                "be positive."
            )

        self.state_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            self.observations_path.with_suffix(
                ".tmp.npz"
            )
        )

        np.savez_compressed(
            temporary,
            d_obs=d_obs,
            Ce=Ce,
        )

        temporary.replace(
            self.observations_path
        )

    def load_observations(self):
        """
        Load the observations used by ES-MDA.

        Returns
        -------
        d_obs : ndarray
            Observed-data vector with shape
            (n_data,).

        Ce : ndarray
            Observation-error covariance matrix with
            shape (n_data, n_data).
        """

        if not self.observations_path.is_file():
            raise FileNotFoundError(
                "Observation checkpoint not found: "
                f"{self.observations_path}"
            )

        try:
            with np.load(
                self.observations_path,
                allow_pickle=False,
            ) as data:

                required = {
                    "d_obs",
                    "Ce",
                }

                if not required.issubset(
                    data.files
                ):
                    raise ValueError(
                        "Observation checkpoint does "
                        "not contain d_obs and Ce."
                    )

                d_obs = np.asarray(
                    data["d_obs"],
                    dtype=float,
                )

                Ce = np.asarray(
                    data["Ce"],
                    dtype=float,
                )

        except (OSError, EOFError) as error:
            raise ValueError(
                "Observation checkpoint could not "
                "be read: "
                f"{self.observations_path}"
            ) from error

        if d_obs.ndim != 1:
            raise ValueError(
                "Checkpoint d_obs must be a "
                "1D array."
            )

        n_data = len(d_obs)

        if Ce.shape != (
            n_data,
            n_data,
        ):
            raise ValueError(
                "Checkpoint Ce must have shape "
                "(n_data, n_data)."
            )

        if not np.all(
            np.isfinite(d_obs)
        ):
            raise ValueError(
                "Checkpoint d_obs contains "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(Ce)
        ):
            raise ValueError(
                "Checkpoint Ce contains "
                "non-finite values."
            )

        if not np.allclose(
            Ce,
            Ce.T,
        ):
            raise ValueError(
                "Checkpoint Ce must be symmetric."
            )

        if np.any(
            np.diag(Ce) <= 0.0
        ):
            raise ValueError(
                "Checkpoint Ce diagonal entries "
                "must be positive."
            )

        return d_obs, Ce

    def save_prior_checkpoint(
        self,
        M,
        D,
        priors,
        realization_ids,
    ):
        """
        Save the initial ensemble state.

        Parameters
        ----------
        M : ndarray
            Model ensemble with shape
            (n_model_parameters, n_ensemble).

        D : ndarray
            Data ensemble with shape
            (n_data, n_ensemble).
    
        priors : dict
            Full-grid prior arrays. Each array must
            have ensemble dimension on the last axis.

        realization_ids : array-like
            Persistent realization IDs.
        """

        M = np.asarray(
            M,
            dtype=float,
        )

        D = np.asarray(
            D,
            dtype=float,
        )

        if D.ndim != 2:
            raise ValueError(
                "D must be a 2D array."
            )

        if D.shape[1] != M.shape[1]:
            raise ValueError(
                "M and D must have the same "
                "ensemble size."
            )

        realization_ids = np.asarray(
            realization_ids,
            dtype=int,
        )

        if M.ndim != 2:
            raise ValueError(
                "M must be a 2D array."
            )

        Ne = M.shape[1]

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

        if len(np.unique(realization_ids)) != Ne:
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

        arrays = {
            "M": M,
            "D": D,
            "realization_ids": realization_ids,
        }

        for variable, prior in priors.items():

            prior = np.asarray(
                prior,
                dtype=float,
            )

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

            if not np.all(np.isfinite(prior)):
                raise ValueError(
                    f"Prior '{variable}' contains "
                    "non-finite values."
                )

            arrays[
                f"prior__{variable}"
            ] = prior

        self.state_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            self.prior_checkpoint_path.with_suffix(
                ".tmp.npz"
            )
        )

        np.savez_compressed(
            temporary,
            **arrays,
        )

        temporary.replace(
            self.prior_checkpoint_path
        )

    def load_prior_checkpoint(self):
        """
        Load the initial ensemble state.

        Returns
        -------
        M : ndarray
            Model ensemble.

        D : ndarray
            Data ensemble.

            Shape must be (n_data, n_ensemble).

        priors : dict
            Full-grid prior arrays.

        realization_ids : ndarray
            Persistent realization IDs.
        """

        checkpoint = self.prior_checkpoint_path

        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Prior checkpoint not found: "
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
                        "Prior checkpoint does not "
                        "contain M, D, and "
                        "realization_ids."
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

                priors = {}

                for name in data.files:
                    if name.startswith(
                        "prior__"
                    ):
                        variable = name[
                            len("prior__"):
                        ]

                        priors[variable] = (
                            np.asarray(
                                data[name],
                                dtype=float,
                            )
                        )

        except (OSError, EOFError) as error:
            raise ValueError(
                "Prior checkpoint could not "
                "be read: "
                f"{checkpoint}"
            ) from error

        if M.ndim != 2:
            raise ValueError(
                "Checkpoint M must be a "
                "2D array."
            )

        Ne = M.shape[1]

        if D.ndim != 2:
            raise ValueError(
                "Checkpoint D must be a "
                "2D array."
            )

        if D.shape[1] != Ne:
            raise ValueError(
                "Checkpoint D does not match "
                "the ensemble size."
            )

        if not np.all(np.isfinite(D)):
            raise ValueError(
                "Checkpoint D contains "
                "non-finite values."
            )

        if realization_ids.ndim != 1:
            raise ValueError(
                "Checkpoint realization_ids "
                "must be a 1D array."
            )

        if len(realization_ids) != Ne:
            raise ValueError(
                "Checkpoint realization_ids "
                "do not match the ensemble size."
            )

        if len(
            np.unique(realization_ids)
        ) != Ne:
            raise ValueError(
                "Checkpoint realization_ids "
                "are not unique."
            )

        if np.any(realization_ids < 1):
            raise ValueError(
                "Checkpoint realization_ids "
                "must be positive."
            )

        if not np.all(np.isfinite(M)):
            raise ValueError(
                "Checkpoint M contains "
                "non-finite values."
            )

        for variable, prior in priors.items():

            if prior.ndim < 1:
                raise ValueError(
                    f"Checkpoint prior "
                    f"'{variable}' must have an "
                    "ensemble dimension."
                )

            if prior.shape[-1] != Ne:
                raise ValueError(
                    f"Checkpoint prior "
                    f"'{variable}' ensemble size "
                    "does not match M."
                )

            if not np.all(
                np.isfinite(prior)
            ):
                raise ValueError(
                    f"Checkpoint prior "
                    f"'{variable}' contains "
                    "non-finite values."
                )

        return M, D, priors, realization_ids
    
    def save_post_checkpoint(
        self,
        M,
        D,
        realization_ids,
    ):
        """
        Save the final posterior ensemble and
        simulated-data ensemble.

        Parameters
        ----------
        M : ndarray
            Final model ensemble with shape
            (n_model_parameters, n_ensemble).

        D : ndarray
            Final simulated-data ensemble with shape
            (n_data, n_ensemble).

        realization_ids : array-like
            Persistent realization IDs.
        """

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

        if M.shape[1] != D.shape[1]:
            raise ValueError(
                "M and D must have the same "
                "ensemble size."
            )

        Ne = M.shape[1]

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

        if len(np.unique(realization_ids)) != Ne:
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

        temporary = (
            self.post_checkpoint_path.with_suffix(
                ".tmp.npz"
            )
        )

        np.savez_compressed(
            temporary,
            M=M,
            D=D,
            realization_ids=realization_ids,
        )

        temporary.replace(
            self.post_checkpoint_path
        )

    def load_post_checkpoint(self):
        """
        Load the final posterior ensemble and
        simulated-data ensemble.

        Returns
        -------
        M : ndarray
            Final model ensemble.

        D : ndarray
            Final simulated-data ensemble.

        realization_ids : ndarray
            Persistent realization IDs.
        """

        checkpoint = self.post_checkpoint_path

        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Post checkpoint not found: "
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
                        "Post checkpoint does not "
                        "contain M, D, and "
                        "realization_ids."
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
                "Post checkpoint could not "
                "be read: "
                f"{checkpoint}"
            ) from error

        if M.ndim != 2:
            raise ValueError(
                "Checkpoint M must be a "
                "2D array."
            )

        if D.ndim != 2:
            raise ValueError(
                "Checkpoint D must be a "
                "2D array."
            )

        if M.shape[1] != D.shape[1]:
            raise ValueError(
                "Checkpoint M and D have "
                "different ensemble sizes."
            )

        Ne = M.shape[1]

        if realization_ids.ndim != 1:
            raise ValueError(
                "Checkpoint realization_ids "
                "must be a 1D array."
            )

        if len(realization_ids) != Ne:
            raise ValueError(
                "Checkpoint realization_ids "
                "do not match the ensemble size."
            )

        if len(
            np.unique(realization_ids)
        ) != Ne:
            raise ValueError(
                "Checkpoint realization_ids "
                "are not unique."
            )

        if np.any(realization_ids < 1):
            raise ValueError(
                "Checkpoint realization_ids "
                "must be positive."
            )

        if not np.all(np.isfinite(M)):
            raise ValueError(
                "Checkpoint M contains "
                "non-finite values."
            )

        if not np.all(np.isfinite(D)):
            raise ValueError(
                "Checkpoint D contains "
                "non-finite values."
            )

        return M, D, realization_ids

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
            or round_number >= self.n_assimilations
        ):
            raise ValueError(
                "round_number must be between 1 and "
                f"{self.n_assimilations - 1}."
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
            self.n_assimilations,
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
            self.n_assimilations,
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
    
    def assimilate(
        self,
        M,
        priors,
        realization_ids,
        forward_model,
        failure_exception,
        d_obs,
        Ce,
        alphas,
        rng,
        inversion="direct",
        energy=0.99,
    ):
        """
        Perform a complete ES-MDA assimilation run.

        The prior ensemble, each pre-update round
        evaluation, and the final posterior evaluation
        are saved as checkpoints.

        Returns
        -------
        EnsembleEvaluation
            Final evaluated posterior ensemble.
        """

        alphas = validate_alphas(alphas)

        if len(alphas) != self.n_assimilations:
            raise ValueError(
                "Number of inflation factors must match "
                "n_assimilations."
            )

        self.create()

        M_current = np.asarray(
            M,
            dtype=float,
        ).copy()

        priors_current = {
            variable: np.asarray(
                prior,
                dtype=float,
            ).copy()
            for variable, prior in priors.items()
        }

        ids_current = np.asarray(
            realization_ids,
            dtype=int,
        ).copy()

        # Evaluate prior
        evaluation = evaluate_ensemble(
            M=M_current,
            priors=priors_current,
            realization_ids=ids_current,
            forward_model=forward_model,
            failure_exception=failure_exception,
        )

        self.save_prior_checkpoint(
            M=evaluation.M,
            D=evaluation.D,
            priors=evaluation.priors,
            realization_ids=(
                evaluation.realization_ids
            ),
        )

        M_current = evaluation.M
        priors_current = evaluation.priors
        ids_current = evaluation.realization_ids



        for assimilation_number, alpha in enumerate(
            alphas,
            start=1,
        ):
            M_updated, _ = esmda_update(
                M=evaluation.M,
                D=evaluation.D,
                d_obs=d_obs,
                Ce=Ce,
                alpha=alpha,
                rng=rng,
                inversion=inversion,
                energy=energy,
            )

            M_current = M_updated
            priors_current = evaluation.priors
            ids_current = evaluation.realization_ids

            if assimilation_number < self.n_assimilations:
                evaluation = evaluate_ensemble(
                    M=M_current,
                    priors=priors_current,
                    realization_ids=ids_current,
                    forward_model=forward_model,
                    failure_exception=failure_exception,
                )

                self.save_round_checkpoint(
                    round_number=assimilation_number,
                    M=evaluation.M,
                    D=evaluation.D,
                    realization_ids=(
                        evaluation.realization_ids
                    ),
                )

        final_evaluation = evaluate_ensemble(
            M=M_current,
            priors=priors_current,
            realization_ids=ids_current,
            forward_model=forward_model,
            failure_exception=failure_exception,
        )

        self.save_post_checkpoint(
            M=final_evaluation.M,
            D=final_evaluation.D,
            realization_ids=(
                final_evaluation.realization_ids
            ),
        )

        return final_evaluation

    def resume(
        self,
        forward_model,
        failure_exception,
        d_obs,
        Ce,
        alphas,
        rng,
        inversion="direct",
        energy=0.99,
    ):
        """
        Resume an interrupted ES-MDA assimilation run.

        Each checkpoint represents an ensemble that has
        already completed its forward evaluation.

        The run resumes from the latest evaluated state,
        performs the next ES-MDA update, and continues
        until the final posterior is evaluated.
        """

        alphas = validate_alphas(alphas)

        if len(alphas) != self.n_assimilations:
            raise ValueError(
                "Number of inflation factors must match "
                "n_assimilations."
            )

        completed = self.completed_rounds()

        # Load the evaluated prior ensemble.
        (
            M_prior,
            D_prior,
            priors_original,
            ids_original,
        ) = self.load_prior_checkpoint()

        if not completed:
            # The prior M0, D0 has already been
            # evaluated. The next operation is
            # ES-MDA update #1.

            M_current = M_prior.copy()
            D_current = D_prior.copy()

            priors_current = {
                name: values.copy()
                for name, values
                in priors_original.items()
            }

            ids_current = ids_original.copy()

            last_evaluated_round = 0

        else:
            # The latest round checkpoint contains
            # an already evaluated Mk, Dk pair.

            last_evaluated_round = completed[-1]

            (
                M_current,
                D_current,
                ids_current,
            ) = self.load_round_checkpoint(
                last_evaluated_round
            )

            # Restore the full-grid priors corresponding
            # to the realizations that survived up to the
            # latest checkpoint.

            id_to_index = {
                realization_id: index
                for index, realization_id
                in enumerate(ids_original)
            }

            try:
                indices = [
                    id_to_index[realization_id]
                    for realization_id
                    in ids_current
                ]
            except KeyError as error:
                raise ValueError(
                    "Round checkpoint contains a "
                    "realization ID that is not present "
                    "in the prior checkpoint."
                ) from error

            priors_current = {
                name: values[..., indices].copy()
                for name, values
                in priors_original.items()
            }

        # Each saved state has already been evaluated.
        #
        # prior       -> next alpha is alphas[0]
        # round_001   -> next alpha is alphas[1]
        # round_002   -> next alpha is alphas[2]
        # ...
        #
        # Therefore last_evaluated_round is also the
        # zero-based index of the next alpha.

        for assimilation_index in range(
            last_evaluated_round,
            self.n_assimilations,
        ):
            M_current, _ = esmda_update(
                M=M_current,
                D=D_current,
                d_obs=d_obs,
                Ce=Ce,
                alpha=alphas[
                    assimilation_index
                ],
                rng=rng,
                inversion=inversion,
                energy=energy,
            )

            # After the final ES-MDA update, M_current
            # is the posterior. It is evaluated below
            # and stored in post.npz.
            if (
                assimilation_index
                == self.n_assimilations - 1
            ):
                break

            # Otherwise evaluate the updated ensemble
            # and save the corresponding intermediate
            # checkpoint.
            evaluation = evaluate_ensemble(
                M=M_current,
                priors=priors_current,
                realization_ids=ids_current,
                forward_model=forward_model,
                failure_exception=failure_exception,
            )

            round_number = (
                assimilation_index + 1
            )

            self.save_round_checkpoint(
                round_number=round_number,
                M=evaluation.M,
                D=evaluation.D,
                realization_ids=(
                    evaluation.realization_ids
                ),
            )

            M_current = evaluation.M
            D_current = evaluation.D
            priors_current = evaluation.priors
            ids_current = (
                evaluation.realization_ids
            )

        # Evaluate the final posterior ensemble.

        final_evaluation = evaluate_ensemble(
            M=M_current,
            priors=priors_current,
            realization_ids=ids_current,
            forward_model=forward_model,
            failure_exception=failure_exception,
        )

        self.save_post_checkpoint(
            M=final_evaluation.M,
            D=final_evaluation.D,
            realization_ids=(
                final_evaluation.realization_ids
            ),
        )

        return final_evaluation