from abc import ABC, abstractmethod


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def _preprocessing_logic():
        pass

    def __call__(self, data):
        return self._preprocessing(data)

    def _preprocessing(self, data_point):
        if isinstance(data_point, tuple):
            data, target = data_point
            data = self._preprocessing_logic(data)
            return data, target
        else:
            data = self._preprocessing_logic(data_point)
            return data
