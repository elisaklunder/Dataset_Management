from abc import ABC, abstractmethod

import numpy as np


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def _preprocessing_logic():
        pass

    def __call__(self, data):
        return self._preprocessing(data)

    def _preprocessing(self, data_point):
        if isinstance(data_point, np.ndarray) or (
            isinstance(data_point, tuple)
            and isinstance(data_point[0], np.ndarray)
            and data_point[0].ndim == 1
        ):
            print("no targett")
            data = self._preprocessing_logic(data_point)
            return data
        else:
            print("targett")
            data, target = data_point
            data = self._preprocessing_logic(data)
            return data, target
