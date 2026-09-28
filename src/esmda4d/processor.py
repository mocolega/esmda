from dataclasses import dataclass

import numpy as np


@dataclass
class StateRequirement:
    """
    Simulator state required by a data processor.

    Parameters
    ----------
    variable
        Name of the simulator state variable.

    dates
        Dates at which the state is required.
    """

    variable: str
    dates: list[np.datetime64]


class DataProcessor:
    """
    Base interface for transforming simulator states
    into data that can be used by ES-MDA.
    """

    def required_states(self):
        """
        Return the simulator states required by
        this processor.
        """
        raise NotImplementedError

    def run(self, states):
        """
        Transform simulator states into a
        one-dimensional data vector.

        Returns
        -------
        numpy.ndarray
            One-dimensional vector containing the
            derived simulated data.
        """
        raise NotImplementedError