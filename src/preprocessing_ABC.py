from abc import ABC


class PreprocessingTechniqueABC(ABC):
    def __call__(self, data):
        return self._preprocessing(data)

    def _preprocessing_logic(self, data):
        pass

    def _preprocessing(self, data_point):
        if isinstance(data_point, tuple):
            data, target = data_point
            data = self._preprocessing_logic(data)
            return data, target
        else:
            data = self._preprocessing_logic(data)
            return data
