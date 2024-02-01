from abc import ABC, abstractmethod

import numpy as np


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def _preprocessing_logic():
        pass

    def __call__(self, data: np.ndarray | tuple) -> np.ndarray | tuple:
        return self._preprocessing(data)

    def _preprocessing(
        self, data_point: np.ndarray | tuple
    ) -> np.ndarray | tuple:
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
