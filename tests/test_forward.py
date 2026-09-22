import numpy as np
import pytest

from esmda4d.forward import ForwardModel


class LinearForwardModel(ForwardModel):
    """
    Simple forward model used only for testing.

    D = G @ M
    """

    def __init__(self, G):
        self.G = np.asarray(
            G,
            dtype=float,
        )

    def run(self, M):
        return self.G @ M


def test_forward_model_run():

    G = np.array([
        [1.0, 0.5],
        [0.2, 1.0],
    ])

    M = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    model = LinearForwardModel(G)

    D = model.run(M)

    expected_D = G @ M

    np.testing.assert_array_equal(
        D,
        expected_D,
    )

    assert D.shape == (2, 3)


def test_forward_model_is_callable():

    G = np.array([
        [1.0, 0.5],
        [0.2, 1.0],
    ])

    M = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    model = LinearForwardModel(G)

    D_run = model.run(M)
    D_call = model(M)

    np.testing.assert_array_equal(
        D_call,
        D_run,
    )


def test_forward_model_requires_run_method():

    class IncompleteForwardModel(ForwardModel):
        pass

    with pytest.raises(
        TypeError,
        match="abstract",
    ):
        IncompleteForwardModel()