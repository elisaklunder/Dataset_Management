from abc import ABC, abstractmethod
from typing import Any


class DataLoaderABC(ABC):
    @abstractmethod
    def _read_data_file() -> Any:
        """
        Abstract method implemented by the concrete class

        Returns:
            Any: the data that has been read
        """
        pass

    def __call__(self, path: str) -> Any:
        """
        Method implementing the object as a callable

        Args:
            path (str): directory given to the loader functions to load
            the data

        Returns:
            Any: the data that has been loaded from the path
        """
        return self._read_data_file(path)
