from abc import ABC, abstractmethod


class BatchLoaderABC(ABC):

    @abstractmethod
    def create_batches(self) -> None:
        """
        abstract method, when is implemented in the concrete class it performs
        the loading of the data in batches.
        """
        pass
