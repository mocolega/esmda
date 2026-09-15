import numpy as np
import pytest

from esmda4d import data_mismatch


def test_zero_mismatch_when_prediction_matches_observation():

    d_obs = np.array([
        1.0,
        2.0,
        3.0,
    ])

    d_pred = d_obs.copy()

    Ce = np.eye(3)

    mismatch = data_mismatch(
        d_pred=d_pred,
        d_obs=d_obs,
        Ce=Ce,
    )

    assert mismatch == 0.0


def test_mismatch_with_identity_covariance():

    d_obs = np.array([
        1.0,
        2.0,
    ])

    d_pred = np.array([
        0.0,
        0.0,
    ])

    Ce = np.eye(2)

    # residual = [1, 2]
    #
    # mismatch =
    # 1 / (2 * Nd) * (1^2 + 2^2)
    #
    # = 1 / 4 * 5
    # = 1.25

    mismatch = data_mismatch(
        d_pred=d_pred,
        d_obs=d_obs,
        Ce=Ce,
    )

    np.testing.assert_allclose(
        mismatch,
        1.25,
    )


def test_mismatch_with_diagonal_covariance():

    d_obs = np.array([
        1.0,
        2.0,
    ])

    d_pred = np.array([
        0.0,
        0.0,
    ])

    sigma = np.array([
        1.0,
        2.0,
    ])

    Ce = np.diag(
        sigma**2
    )

    # residual = [1, 2]
    #
    # normalized residuals:
    #
    # 1 / 1 = 1
    # 2 / 2 = 1
    #
    # mismatch =
    # 1 / (2 * 2) * (1^2 + 1^2)
    #
    # = 0.5

    mismatch = data_mismatch(
        d_pred=d_pred,
        d_obs=d_obs,
        Ce=Ce,
    )

    np.testing.assert_allclose(
        mismatch,
        0.5,
    )


def test_mismatch_with_correlated_errors():

    d_obs = np.array([
        1.0,
        2.0,
    ])

    d_pred = np.array([
        0.0,
        0.0,
    ])

    Ce = np.array([
        [1.0, 0.5],
        [0.5, 2.0],
    ])

    residual = d_obs - d_pred

    expected = (
        residual.T
        @ np.linalg.solve(Ce, residual)
    ) / (2 * len(d_obs))

    mismatch = data_mismatch(
        d_pred=d_pred,
        d_obs=d_obs,
        Ce=Ce,
    )

    np.testing.assert_allclose(
        mismatch,
        expected,
    )


def test_d_pred_must_be_1d():

    d_pred = np.array([
        [1.0],
        [2.0],
    ])

    d_obs = np.array([
        1.0,
        2.0,
    ])

    Ce = np.eye(2)

    with pytest.raises(
        ValueError,
        match="d_pred must be a 1D array",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_d_obs_must_be_1d():

    d_pred = np.array([
        1.0,
        2.0,
    ])

    d_obs = np.array([
        [1.0],
        [2.0],
    ])

    Ce = np.eye(2)

    with pytest.raises(
        ValueError,
        match="d_obs must be a 1D array",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_d_pred_and_d_obs_must_have_same_shape():

    d_pred = np.array([
        1.0,
        2.0,
    ])

    d_obs = np.array([
        1.0,
        2.0,
        3.0,
    ])

    Ce = np.eye(3)

    with pytest.raises(
        ValueError,
        match="d_pred and d_obs must have the same shape",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_Ce_shape_must_match_number_of_data():

    d_pred = np.array([
        1.0,
        2.0,
    ])

    d_obs = np.array([
        1.0,
        2.0,
    ])

    Ce = np.eye(3)

    with pytest.raises(
        ValueError,
        match="Ce must have shape",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_d_pred_cannot_contain_nan():

    d_pred = np.array([
        np.nan,
        2.0,
    ])

    d_obs = np.array([
        1.0,
        2.0,
    ])

    Ce = np.eye(2)

    with pytest.raises(
        ValueError,
        match="d_pred contains NaN or infinite values",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_d_obs_cannot_contain_infinity():

    d_pred = np.array([
        1.0,
        2.0,
    ])

    d_obs = np.array([
        1.0,
        np.inf,
    ])

    Ce = np.eye(2)

    with pytest.raises(
        ValueError,
        match="d_obs contains NaN or infinite values",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )


def test_Ce_cannot_contain_nan():

    d_pred = np.array([
        1.0,
        2.0,
    ])

    d_obs = np.array([
        1.0,
        2.0,
    ])

    Ce = np.eye(2)
    Ce[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="Ce contains NaN or infinite values",
    ):
        data_mismatch(
            d_pred=d_pred,
            d_obs=d_obs,
            Ce=Ce,
        )