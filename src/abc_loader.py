from abc import ABC, abstractmethod

class DataLoaderABC(ABC):
    @abstractmethod
    def _read_data_file():
        pass

    def __call__(self, path):
        return self._read_data_file(path)
