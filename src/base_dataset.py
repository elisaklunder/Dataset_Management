import copy
import csv
import os
import os.path
import random
from typing import List, TypeVar
from errors import Errors

Dataset = TypeVar("Dataset", bound="BaseDataset")


class BaseDataset:
    def __init__(self) -> None:
        """
        Constructor of the class.
        """
        self._root = None
        self._labels_path = None
        self._format = None
        self._strategy = None
        self._data = []
        self._targets = []
        self.errors = Errors()

    @property
    def data(self) -> List:
        """
        Getter for the attribute 'data'.

        Returns:
            List: deepcopy of the attribute 'data'. In this way 'data'
            is returned safely.
        """
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, new_data: List) -> None:
        """
        Setter for the attribute 'data'.

        Args:
            new_data (List): new value that the attribute 'data' will have.
        """
        self.errors.type_check("new_data", new_data, list)
        self._data = new_data

    @property
    def targets(self) -> List:
        """
        Getter for the attribute 'target'.

        Returns:
            List: deepcopy of the attribute 'target'. In this way 'target'
            is returned safely.
        """
        return copy.deepcopy(self._targets)

    @targets.setter
    def targets(self, new_targets: List) -> None:
        """
        Setter for the attribute 'target'.

        Args:
            new_targets (List): new value that the attribute 'target'
            will have.
        """
        self.errors.type_check("new_targets", new_targets, list)
        self._targets = new_targets

    def _read_data_file(self) -> None:
        """
        Method that is be implemented in the respective subclasses

        Raises:
            NotImplementedError: the method is not implemented and
            should not be used when accessed directly from thid object
        """
        raise NotImplementedError(
            "Method to read in the data needs to be implemented"
        )

    def _csv_to_labels(self) -> None:
        """
        The method sets the 'data' attribute so that the file name is
        not stored anymore.

        If there is a csv, the method reads in the csv file and orders the
        data in such a way that the target for the data[index] will be
        target[index].
        """
        if self._labels_path is not None:
            with open(self._labels_path, "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    self._targets.append(row)
            data = []
            data_dict = {filename: data for data, filename in self.data}
            for index, target in enumerate(self._targets):
                if index != 0:  # because first row contains names of columns
                    data.append(data_dict[target[1]])
            self.data = copy.deepcopy(data)
            self.targets = [item[2] for item in self.targets]
        else:
            self.data = [image for image, _ in self.data]

    def _csv_load_data(self) -> None:
        """
        The method stores the data depending on how the user wants to load it.
        In case of eager loading the data stores a list of data points,
        otherwise the data list would consist of directories.
        """
        for filename in os.listdir(self._root):
            path = os.path.join(self._root, filename)
            if self._strategy == "lazy":
                self._data.append((path, filename))
            elif self._strategy == "eager":
                image = self._read_data_file(path)
                self._data.append((image, filename))
        self._csv_to_labels()

    def load_data(
        self,
        root: str,
        strategy: str,
        format: str = "csv",
        labels_path: str = None,
    ) -> None:
        """
        Public method calling helper functions defined above to correctly load
        the data into the program.

        Args:
            root (str): _description_
            strategy (str): _description_
            format (str, optional): _description_. Defaults to "csv".
            labels_path (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.errors.type_check("root", root, str)
        self.errors.type_check("strategy", strategy, str)
        self.errors.value_check("strategy", strategy, "lazy", "eager")
        self.errors.type_check("format", format, str)
        self.errors.value_check("format", format, "csv", "hierarchical")
        if format != "csv" and labels_path is not None:
            raise ValueError(
                "Labels path should only be provided when the format is 'csv'."
            )
        self.errors.type_check("labels_path", labels_path, str, type(None))

        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if self._format == "csv":
            self._csv_load_data()

    def _get_item_format_helper(self, data: List, index: int) -> List | tuple:
        if bool(self._labels_path) or self._format == "hierarchical":
            return data, self.targets[index]
        else:
            return data

    def __getitem__(self, index: int) -> List | tuple:
        if not bool(self._data):
            raise ValueError("No data available.")
        self.errors.type_check("index", index, int)
        try:
            if self._strategy == "eager":
                data = self._data[index]
            elif self._strategy == "lazy":
                file_dir = self._data[index]
                data = self._read_data_file(file_dir)
            return self._get_item_format_helper(data, index)
        except IndexError:
            raise IndexError("Index out of range.")

    def __len__(self) -> int:
        return len(self._data)

    def train_test_split(
        self, train_size: float, shuffle: bool
    ) -> Dataset:
        if not bool(self._data):
            raise ValueError(
                "Unable to perform a train-test split because no data \
has been loaded in the dataset yet."
            )
        self.errors.type_check("shuffle", shuffle, bool)
        self.errors.type_check("train_size", train_size, float)

        data_len = len(self)
        train_len = int(data_len * train_size)

        train = copy.deepcopy(self)
        test = copy.deepcopy(self)

        # using getters here to copy the data and targets to local variables
        # to make sure that the original dataset won't be unintentionally
        # shuffled when performing the split
        data = self.data
        targets = self.targets

        if shuffle:
            if bool(self._targets):
                zipped_data = list(zip(data, targets))
                shuffled_data = random.sample(zipped_data, data_len)
                data, targets = map(list, zip(*shuffled_data))
            else:
                data = random.sample(data, data_len)

        train.data = data[:train_len]
        test.data = data[train_len::]

        if bool(self._targets):
            train.targets = targets[:train_len]
            test.targets = targets[train_len::]

        return train, test
