from abc import ABC, abstractmethod
import numpy as np

class ForwardModel(ABC):
    """
    Base interface for ES-MDA forward models.

    A forward model receives a model ensemble M and
    returns the corresponding predicted-data ensemble D.

    Convention
    ----------
    M.shape = (n_model_parameters, n_ensemble)
    D.shape = (n_data, n_ensemble)
    """

    def __call__(self, M):
        return self.run(M)
    
    @abstractmethod
    def run(self, M):
        """
        Evaluate the forward model for ensemble M.

        Parameters
        ----------
        M : ndarray
            Model ensemble with shape
            (n_model_parameters, n_ensemble).

        Returns
        -------
        D : ndarray
            Predicted-data ensemble with shape
            (n_data, n_ensemble).
        """
        pass