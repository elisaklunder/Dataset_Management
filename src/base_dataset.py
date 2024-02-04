import copy
import csv
import os
import os.path
import random
from typing import List, TypeVar

from errors import Errors
from src.abc_base_dataset import BaseDatasetABC

Dataset = TypeVar("Dataset", bound="BaseDataset")


class BaseDataset(BaseDatasetABC):
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
        self._errors = Errors()

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

        Raises:
            TypeError: if new_data argument is not of the type list
        """
        self._errors.type_check("new_data", new_data, list)
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

        Raises:
            TypeError: if new_trgets argument is not of the type list
        """
        self._errors.type_check("new_targets", new_targets, list)
        self._targets = new_targets

    def _read_data_file(self) -> None:
        """
        Method that is be implemented in the respective subclasses.

        Raises:
            NotImplementedError: the method is not implemented and
            should not be used when accessed directly from this object.
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
                temp_targets = []
                for row in csv_reader:
                    temp_targets.append(row)
                self.targets = temp_targets
            data_dict = {filename: data for data, filename in self.data}
            temp_data = []
            for index, target in enumerate(self.targets):
                if index != 0:  # because first row contains names of columns
                    temp_data.append(data_dict[target[1]])
            self.data = copy.deepcopy(temp_data)
            self.targets = [item[2] for item in self.targets]
        else:
            self.data = [image for image, _ in self.data]

    def _csv_load_data(self) -> None:
        """
        The method stores the data depending on how the user wants to load it.
        In case of eager loading the data stores a list of data points,
        otherwise the data list would consist of directories.
        """
        temp_data = []
        for filename in os.listdir(self._root):
            path = os.path.join(self._root, filename)
            if self._strategy == "lazy":
                temp_data.append((path, filename))
            elif self._strategy == "eager":
                image = self._read_data_file(path)
                temp_data.append((image, filename))
        self.data = temp_data
        self._csv_to_labels()

    def load_data(
        self,
        root: str,
        strategy: str,
        format: str,
        labels_path: str = None,
    ) -> None:
        """
        Public method calling helper functions defined above to correctly load
        the data into the program.

        Args:
            root (str): directory indicating a path to the data to be loaded.

            strategy (str): string specifying whether the data is loaded in a
            lazy or eager fashion.

            format (str, optional): string indicating the structure of the
            data. Defaults to "csv".

            labels_path (str, optional): directory indicating the path to the
            labels, if any. Defaults to None.

        Raises:
            TypeError: if the root argument is not a string.
            TypeError: if the strategy argument is not a string.
            TypeError: if the format argument is not a string.
            TypeError: if the labels_path argument is not a string or None.
            ValeError: if the strategy argument is not one of the expected
            strings.
            ValueError: if the format argument is not one of the expected
            strings
            ValueError: if the specified structure is hierarchical but the
            user has specified a path for the csv file.
        """
        self._errors.type_check("root", root, str)
        self._errors.type_check("strategy", strategy, str)
        self._errors.value_check("strategy", strategy, "lazy", "eager")
        self._errors.type_check("format", format, str)
        self._errors.value_check("format", format, "csv", "hierarchical")
        if format != "csv" and labels_path is not None:
            raise ValueError(
                "Labels path should only be provided when the format is 'csv'."
            )
        self._errors.type_check("labels_path", labels_path, str, type(None))

        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if self._format == "csv":
            self._csv_load_data()

    def _get_item_format_helper(self, data: List, index: int) -> List | tuple:
        """
        Helper private method for __getitem__.

        Args:
            data (List): List containing all the data points.
            index (int): index to be accessed with subsetting operator.

        Returns:
            List | tuple: values containing the data or the data and the target
            at a given index.
        """
        if bool(self._labels_path) or self._format == "hierarchical":
            return data, self.targets[index]
        else:
            return data

    def __getitem__(self, index: int) -> List | tuple:
        """
        Megic method to implement the subsetting operator for the class.

        Args:
            index (int): integer value indicating the index to be accessed with
            subsetting operator.

        Raises:
            ValueError: if there is no data contained in the self.data
            attribute, namely the data has not been loaded or is an empty list.
            TypeError: if the index argument is not an int.
            IndexError: if the index is out of the range of the 'data'
            attribute.

        Returns:
            List | tuple: _description_
        """
        if not bool(self.data):
            raise ValueError("No data available.")
        self._errors.type_check("index", index, int)
        try:
            if self._strategy == "eager":
                data = self.data[index]
            elif self._strategy == "lazy":
                file_dir = self.data[index]
                data = self._read_data_file(file_dir)
            return self._get_item_format_helper(data, index)
        except IndexError:
            raise IndexError("Index out of range.")

    def __len__(self) -> int:
        """
        Magic method implementing the len() function for the object.

        Returns:
            int: integer number indicating the number the number of batches
            that can be created from the dataset with the specific batch size.
        """
        return len(self.data)

    def train_test_split(self, train_size: float, shuffle: bool) -> Dataset:
        """
        Public method performing the train test split.

        Args:
            train_size (float): floating number indicating the percentage of
            the data in the train.

            shuffle (bool):Bool for expressing whether the data needs to be
            shuffled before performing the train test split.

        Raises:
            ValueError: if there is no data contained in the self.data
            attribute, namely the data has not been loaded or is an empty list.
            TypeError: if the suffle argument is not a boolean value.
            TypeError: if the train_size argument is not a float.

        Returns:
            Dataset: one Dataset object for the train and one for the test.
        """
        if not bool(self.data):
            raise ValueError(
                "Unable to perform a train-test split because no data \
has been loaded in the dataset yet."
            )
        self._errors.type_check("shuffle", shuffle, bool)
        self._errors.type_check("train_size", train_size, float)

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
            if bool(self.targets):
                zipped_data = list(zip(data, targets))
                shuffled_data = random.sample(zipped_data, data_len)
                data, targets = map(list, zip(*shuffled_data))
            else:
                data = random.sample(data, data_len)

        train.data = data[:train_len]
        test.data = data[train_len::]

        if bool(self.targets):
            train.targets = targets[:train_len]
            test.targets = targets[train_len::]

        return train, test
