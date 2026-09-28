from esmda4d.cmg.validation import (
    CMGRunValidator,
)


def test_cmg_validator_accepts_normal_termination(
    tmp_path,
):
    model_path = tmp_path / "model_0001.dat"
    log_path = tmp_path / "model_0001.log"

    model_path.write_text("CMG model")

    log_path.write_text(
        """
        Date and Time of Start of Run

        End of Simulation: Normal Termination

        Date and Time of End of Run
        """
    )

    validator = CMGRunValidator()

    result = validator.validate(
        model_path
    )

    assert result.valid
    assert result.reason is None
    assert result.log_path == log_path

def test_cmg_validator_rejects_abnormal_termination(
    tmp_path,
):
    model_path = tmp_path / "model_0002.dat"
    log_path = tmp_path / "model_0002.log"

    model_path.write_text("CMG model")

    log_path.write_text(
        """
        FATAL ERROR

        Terminating simulation: Fatal error.

        End of Simulation: Abnormal Termination
        """
    )

    validator = CMGRunValidator()

    result = validator.validate(
        model_path
    )

    assert not result.valid
    assert result.reason == (
        "CMG simulation did not end "
        "with normal termination."
    )

def test_cmg_validator_rejects_incomplete_log(
    tmp_path,
):
    model_path = tmp_path / "model_0003.dat"
    log_path = tmp_path / "model_0003.log"

    model_path.write_text("CMG model")

    log_path.write_text(
        """
        STOP : Unable to open the following
        file as include file.

        Cannot find include file.
        """
    )

    validator = CMGRunValidator()

    result = validator.validate(
        model_path
    )

    assert not result.valid

def test_cmg_validator_rejects_missing_log(
    tmp_path,
):
    model_path = tmp_path / "model_0004.dat"

    model_path.write_text("CMG model")

    validator = CMGRunValidator()

    result = validator.validate(
        model_path
    )

    assert not result.valid

    assert result.reason == (
        "CMG log file not found."
    )

    assert (
        result.log_path
        == tmp_path / "model_0004.log"
    )