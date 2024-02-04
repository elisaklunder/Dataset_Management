from abc import ABC, abstractmethod

import numpy as np


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def _preprocessing_logic():
        """
        Abstract method that impleents the logic of the preprocessing step to
        be taken.
        """
        pass

    def __call__(self, data: np.ndarray | tuple) -> np.ndarray | tuple:
        """
        Magic method implementing the call functionality for the given object.

        Args:
            data (np.ndarray | tuple): data to perform the preprocessing step
            on.

        Returns:
            np.ndarray | tuple: processed data.
        """
        return self._preprocessing(data)

    def _preprocessing(
        self, data_point: np.ndarray | tuple
    ) -> np.ndarray | tuple:
        """
        Concrete method handles whether the data has labels or not.

        Args:
            data_point (np.ndarray | tuple): data to perform preprocessing
            steps on.

        Returns:
            np.ndarray | tuple: processed data, in tha same format as the
            original one.
        """
        if isinstance(data_point, np.ndarray) or (
            isinstance(data_point, tuple)
            and isinstance(data_point[0], np.ndarray)
            and data_point[0].ndim == 1
        ):
            data = self._preprocessing_logic(data_point)
            return data
        else:
            data, target = data_point
            data = self._preprocessing_logic(data)
            return data, target
