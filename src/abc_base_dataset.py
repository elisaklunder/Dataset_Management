from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar

BaseDatasetABC = TypeVar("BaseDatasetABC", bound="BaseDatasetABC")


class BaseDatasetABC(ABC):
    @abstractmethod
    def load_data(self) -> None:
        """
        Abstract method that will load the data once it is implemented in the
        as a concrete class.
        """
        pass

    @abstractmethod
    def train_test_split(
        self, train_size: float, shuffle: bool
    ) -> BaseDatasetABC:
        """
        Abstract method for performing the train test split.

        Args:
            train_size (flaot): floating number indicating the percentage of
            the data in the train.
            shuffle (bool): Bool for expressing whether the data needs to be
            shuffled before performing the train test split
        """
        pass

    @abstractproperty
    def data():
        pass

    @data.setter
    def data():
        pass
