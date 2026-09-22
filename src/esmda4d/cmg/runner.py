from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass
class SimulationResult:
    """
    Result of one reservoir simulation.
    """

    model_path: Path
    return_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self):
        return self.return_code == 0


class LocalCMGRunner:
    """
    Run CMG simulation models on the local computer.
    """

    def __init__(
        self,
        executable,
        arguments=None,
    ):
        self.executable = Path(
            executable
        )

        if arguments is None:
            arguments = ["{model}"]

        self.arguments = list(
            arguments
        )

        self._validate()

    def _validate(self):

        if not self.executable.is_file():
            raise FileNotFoundError(
                f"CMG executable not found: "
                f"{self.executable}"
            )
    def _build_command(
        self,
        model_path,
    ):

        arguments = [
            argument.replace(
                "{model}",
                model_path.name,
            )
            for argument in self.arguments
        ]

        return [
            str(self.executable),
            *arguments,
        ]
    def run(self, model_path):
        """
        Run one CMG model.

        Parameters
        ----------
        model_path : str or Path
            Path to the CMG .dat file.

        Returns
        -------
        SimulationResult
            Information about the completed simulation.
        """

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"CMG model not found: "
                f"{model_path}"
            )

        command = self._build_command(
            model_path
        )

        process = subprocess.run(
            command,
            cwd=model_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        return SimulationResult(
            model_path=model_path,
            return_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def run_ensemble(
        self,
        model_paths,
    ):
        """
        Run an ensemble of CMG models sequentially.

        Parameters
        ----------
        model_paths : iterable of str or Path
            Paths to the CMG .dat files.

        Returns
        -------
        results : list of SimulationResult
            Simulation results in the same order as
            model_paths.
        """

        model_paths = list(model_paths)

        if len(model_paths) == 0:
            raise ValueError(
                "model_paths cannot be empty."
            )

        results = []

        for model_path in model_paths:

            result = self.run(
                model_path
            )

            results.append(
                result
            )

        return results    