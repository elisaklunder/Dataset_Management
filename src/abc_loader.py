from abc import ABC, abstractmethod


class DataLoaderABC(ABC):
    @abstractmethod
    def _read_data_file():
        """
        Method implemented by the concrete class
        """
        pass

    def __call__(self, path: str) -> None:
        """
        Method implementing the object as a callable

        Args:
            path (str): directory given to the loader functions to load
            the data

        Returns:
            None
        """
        return self._read_data_file(path)
