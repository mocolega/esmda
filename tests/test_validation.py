import numpy as np
import pytest

from esmda4d import esmda_update


def build_valid_inputs():
    """
    Create a small valid ES-MDA problem that can be modified
    by individual validation tests.
    """

    rng = np.random.default_rng(1234)

    Nm = 3
    Nd = 4
    Ne = 10

    M = rng.normal(size=(Nm, Ne))
    D = rng.normal(size=(Nd, Ne))

    d_obs = rng.normal(size=Nd)

    Ce = np.eye(Nd)

    alpha = 4.0

    return M, D, d_obs, Ce, alpha, rng


def test_valid_inputs_do_not_raise():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    esmda_update(
        M=M,
        D=D,
        d_obs=d_obs,
        Ce=Ce,
        alpha=alpha,
        rng=rng,
        inversion="direct",
    )


def test_M_must_be_2d():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    M = M[0]

    with pytest.raises(
        ValueError,
        match="M must be a 2D array",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_D_must_be_2d():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    D = D[0]

    with pytest.raises(
        ValueError,
        match="D must be a 2D array",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_d_obs_must_be_1d():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    d_obs = d_obs[:, None]

    with pytest.raises(
        ValueError,
        match="d_obs must be a 1D array",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_Ce_must_be_2d():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    Ce = np.diag(Ce)

    with pytest.raises(
        ValueError,
        match="Ce must be a 2D covariance matrix",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_ensemble_must_have_at_least_two_members():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    M = M[:, :1]
    D = D[:, :1]

    with pytest.raises(
        ValueError,
        match="at least 2 realizations",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_M_and_D_must_have_same_ensemble_size():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    D = D[:, :-1]

    with pytest.raises(
        ValueError,
        match="same number of ensemble realizations",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_d_obs_length_must_match_D():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    d_obs = d_obs[:-1]

    with pytest.raises(
        ValueError,
        match="d_obs length must match",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_Ce_shape_must_match_number_of_data():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    Ce = np.eye(3)

    with pytest.raises(
        ValueError,
        match="Ce must have shape",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_alpha_must_be_positive():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    alpha = 0.0

    with pytest.raises(
        ValueError,
        match="alpha must be positive",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -1.0, -4.0],
)
def test_invalid_alpha_values(invalid_alpha):

    M, D, d_obs, Ce, _, rng = build_valid_inputs()

    with pytest.raises(
        ValueError,
        match="alpha must be positive",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=invalid_alpha,
            rng=rng,
        )


def test_M_cannot_contain_nan():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    M[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="M contains NaN or infinite values",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_D_cannot_contain_infinity():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    D[0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="D contains NaN or infinite values",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_d_obs_cannot_contain_nan():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    d_obs[0] = np.nan

    with pytest.raises(
        ValueError,
        match="d_obs contains NaN or infinite values",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_Ce_cannot_contain_infinity():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    Ce[0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="Ce contains NaN or infinite values",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )

def test_Ce_must_be_symmetric():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    Ce[0, 1] = 0.5
    Ce[1, 0] = 0.0

    with pytest.raises(
        ValueError,
        match="Ce must be symmetric",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )


def test_Ce_diagonal_must_be_positive():

    M, D, d_obs, Ce, alpha, rng = build_valid_inputs()

    Ce[0, 0] = 0.0

    with pytest.raises(
        ValueError,
        match="Ce diagonal entries must be positive",
    ):
        esmda_update(
            M=M,
            D=D,
            d_obs=d_obs,
            Ce=Ce,
            alpha=alpha,
            rng=rng,
        )