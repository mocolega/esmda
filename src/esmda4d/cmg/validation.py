from dataclasses import dataclass
from pathlib import Path


NORMAL_TERMINATION = (
    "End of Simulation: Normal Termination"
)


@dataclass
class CMGRunValidation:
    """
    Result of validating a CMG simulation log.
    """

    valid: bool
    log_path: Path
    reason: str | None = None


class CMGRunValidator:
    """
    Validate whether a CMG simulation completed
    normally.
    """

    def validate(self, model_path):
        model_path = Path(model_path)

        log_path = model_path.with_suffix(".log")

        if not log_path.exists():
            return CMGRunValidation(
                valid=False,
                log_path=log_path,
                reason="CMG log file not found.",
            )

        try:
            text = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            return CMGRunValidation(
                valid=False,
                log_path=log_path,
                reason=(
                    "CMG log file could not "
                    f"be read: {error}"
                ),
            )

        if NORMAL_TERMINATION not in text:
            return CMGRunValidation(
                valid=False,
                log_path=log_path,
                reason=(
                    "CMG simulation did not end "
                    "with normal termination."
                ),
            )

        return CMGRunValidation(
            valid=True,
            log_path=log_path,
        )