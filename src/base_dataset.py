import copy
import csv
import os
import os.path
import random


class BaseDataset:
    def __init__(self):
        self._root = None
        self._labels_path = None
        self._format = None
        self._strategy = None
        self._data = []
        self._targets = []

    def _type_check(self, argument_name, argument, *expected_types):
        if type(argument) not in expected_types:
            types_str = " or ".join(str(type) for type in expected_types)
            raise TypeError(
                f"Invalid type for argument '{argument_name}'. Expected \
{types_str}, but got {type(argument)} instead."
            )

    def _value_check(self, argument_name, argument, *accepted_arguments):
        if argument not in accepted_arguments:
            raise ValueError(
                f"Input '{argument}' for argument '{argument_name}' is not \
valid. Expected one of the following: {accepted_arguments}."
            )

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, new_data):
        self._type_check("new_data", new_data, list)
        self._data = new_data

    @property
    def targets(self):
        return copy.deepcopy(self._targets)

    @targets.setter
    def targets(self, new_targets):
        self._type_check("new_data", new_targets, list)
        self._targets = new_targets

    def _read_data_file(self):
        raise NotImplementedError(
            "Method to read in the data needs to be implemented"
        )

    def _csv_to_labels(self):
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

    def _csv_load_data(self):
        for filename in os.listdir(self._root):
            path = os.path.join(self._root, filename)
            if self._strategy == "lazy":
                self._data.append((path, filename))
            elif self._strategy == "eager":
                image = self._read_data_file(path)
                self._data.append((image, filename))
        self._csv_to_labels()

    def load_data(self, root, strategy, format="csv", labels_path=None):
        self._type_check("root", root, str)
        self._type_check("strategy", strategy, str)
        self._value_check("strategy", strategy, "lazy", "eager")
        self._type_check("format", format, str)
        self._value_check("format", format, "csv", "hierarchical")
        if format != "csv" and labels_path is not None:
            raise ValueError(
                "Labels path should only be provided when the format is 'csv'."
            )
        self._type_check("labels_path", labels_path, str, type(None))

        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if self._format == "csv":
            self._csv_load_data()

    def _get_item_format_helper(self, data, index):
        if bool(self._labels_path) or self._format == "hierarchical":
            return data, self.targets[index]
        else:
            return data

    def __getitem__(self, index: int):
        if not bool(self._data):
            raise IndexError("No data available.")
        if not isinstance(index, int):
            raise TypeError("Invalid type for the index. Expected an integer.")
        try:
            if self._strategy == "eager":
                data = self._data[index]
            elif self._strategy == "lazy":
                file_dir = self._data[index]
                data = self._read_data_file(file_dir)
            return self._get_item_format_helper(data, index)
        except IndexError:
            raise IndexError("Index out of range.")

    def __len__(self):
        return len(self._data)

    def train_test_split(self, train_size, shuffle):
        if not bool(self._data):
            raise ValueError(
                "Unable to perform a train-test split because no data \
has been loaded in the dataset yet."
            )

        data_len = len(self._data)  # MAYBE DO LEN(SELF)
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
