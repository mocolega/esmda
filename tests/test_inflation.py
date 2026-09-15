import numpy as np
import pytest

from esmda4d import validate_alphas


def test_valid_constant_alphas():

    alphas = validate_alphas(
        [4.0, 4.0, 4.0, 4.0]
    )

    np.testing.assert_allclose(
        alphas,
        np.array([4.0, 4.0, 4.0, 4.0]),
    )


def test_invalid_alphas():

    with pytest.raises(ValueError):
        validate_alphas(
            [2.0, 2.0, 2.0, 2.0]
        )


def test_negative_alpha():

    with pytest.raises(ValueError):
        validate_alphas(
            [4.0, 4.0, -4.0, 4.0]
        )