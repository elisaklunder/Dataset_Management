from abc import ABC, abstractproperty, abstractmethod


class BaseDatasetABC(ABC):

    @abstractmethod
    def load_data(self, root: str) -> None:
        # root maybe not
        """
        Abstract method that will load the data once it is implemented in the
        as a concrete class.

        Args:
            root (str): directory indicating a path to the data to be loaded
        """
        pass

    @abstractmethod
    def train_test_split(self, train_size: float, shuffle: bool):
        # idk the type hints for this shit
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
