import sys

from esmda4d.cmg.runner import LocalCMGRunner


def test_local_cmg_runner_success(tmp_path):
    """
    Test a successful local simulation.
    """

    # Create fake realization directory
    realization_dir = (
        tmp_path
        / "realization_0001"
    )

    realization_dir.mkdir()

    # Create fake CMG model
    model_path = (
        realization_dir
        / "model_0001.dat"
    )

    model_path.write_text(
        "FAKE CMG MODEL"
    )

    # Use Python itself as our fake simulator.
    #
    # The command will effectively be:
    #
    # python -c "..." model_0001.dat

    code = (
        "import sys; "
        "from pathlib import Path; "
        "model = Path(sys.argv[1]); "
        "print(f'Running {model.name}'); "
        "print(f'Working directory: {Path.cwd().name}')"
    )

    runner = LocalCMGRunner(
        executable=sys.executable,
        arguments=[
            "-c",
            code,
            "{model}",
        ],
    )

    result = runner.run(
        model_path
    )

    assert result.succeeded
    assert result.return_code == 0

    assert (
        "Running model_0001.dat"
        in result.stdout
    )

    assert (
        "Working directory: realization_0001"
        in result.stdout
    )

    assert result.stderr == ""


def test_local_cmg_runner_failure(tmp_path):
    """
    Test detection of a failed simulation.
    """

    realization_dir = (
        tmp_path
        / "realization_0001"
    )

    realization_dir.mkdir()

    model_path = (
        realization_dir
        / "model_0001.dat"
    )

    model_path.write_text(
        "FAKE CMG MODEL"
    )

    code = (
        "import sys; "
        "print('Simulation failed', file=sys.stderr); "
        "sys.exit(7)"
    )

    runner = LocalCMGRunner(
        executable=sys.executable,
        arguments=[
            "-c",
            code,
            "{model}",
        ],
    )

    result = runner.run(
        model_path
    )

    assert not result.succeeded
    assert result.return_code == 7

    assert (
        "Simulation failed"
        in result.stderr
    )


def test_local_cmg_runner_rejects_missing_model(
    tmp_path,
):
    """
    A missing .dat file should raise an error before
    trying to launch the simulator.
    """

    runner = LocalCMGRunner(
        executable=sys.executable,
    )

    missing_model = (
        tmp_path
        / "model_0001.dat"
    )

    try:
        runner.run(
            missing_model
        )

    except FileNotFoundError as error:
        assert "CMG model not found" in str(error)

    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )


def test_local_cmg_runner_rejects_missing_executable(
    tmp_path,
):
    """
    A nonexistent simulator executable should be
    rejected when creating the runner.
    """

    missing_executable = (
        tmp_path
        / "cmg_not_here.exe"
    )

    try:
        LocalCMGRunner(
            executable=missing_executable,
        )

    except FileNotFoundError as error:
        assert "CMG executable not found" in str(error)

    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )


def test_local_cmg_runner_runs_ensemble_sequentially(
        tmp_path,
    ):
        """
        Test sequential execution of several models.
        """

        model_paths = []

        for i in range(1, 4):

            realization_id = f"{i:04d}"

            realization_dir = (
                tmp_path
                / f"realization_{realization_id}"
            )

            realization_dir.mkdir()

            model_path = (
                realization_dir
                / f"model_{realization_id}.dat"
            )

            model_path.write_text(
                "FAKE CMG MODEL"
            )

            model_paths.append(
                model_path
            )

        code = (
            "import sys; "
            "from pathlib import Path; "
            "model = Path(sys.argv[1]); "
            "print(f'Running {model.name}')"
        )

        runner = LocalCMGRunner(
            executable=sys.executable,
            arguments=[
                "-c",
                code,
                "{model}",
            ],
        )

        results = runner.run_ensemble(
            model_paths
        )

        assert len(results) == 3

        for i, result in enumerate(
            results,
            start=1,
        ):

            realization_id = f"{i:04d}"

            assert result.succeeded

            assert (
                result.model_path.name
                == f"model_{realization_id}.dat"
            )

            assert (
                f"Running model_{realization_id}.dat"
                in result.stdout
            )

def test_local_cmg_runner_continues_after_failure(
        tmp_path,
    ):
        """
        One failed realization should not prevent later
        realizations from running.
        """

        model_paths = []

        for i in range(1, 4):

            realization_id = f"{i:04d}"

            realization_dir = (
                tmp_path
                / f"realization_{realization_id}"
            )

            realization_dir.mkdir()

            model_path = (
                realization_dir
                / f"model_{realization_id}.dat"
            )

            model_path.write_text(
                "FAKE CMG MODEL"
            )

            model_paths.append(
                model_path
            )

        code = (
            "import sys; "
            "from pathlib import Path; "
            "model = Path(sys.argv[1]); "
            "print(f'Running {model.name}'); "
            "sys.exit(5 if '0002' in model.name else 0)"
        )

        runner = LocalCMGRunner(
            executable=sys.executable,
            arguments=[
                "-c",
                code,
                "{model}",
            ],
        )

        results = runner.run_ensemble(
            model_paths
        )

        assert len(results) == 3

        assert results[0].succeeded

        assert not results[1].succeeded
        assert results[1].return_code == 5

        assert results[2].succeeded

def test_local_cmg_runner_rejects_empty_ensemble():

        runner = LocalCMGRunner(
            executable=sys.executable,
        )

        try:
            runner.run_ensemble([])

        except ValueError as error:
            assert (
                "model_paths cannot be empty"
                in str(error)
            )

        else:
            raise AssertionError(
                "Expected ValueError."
            )