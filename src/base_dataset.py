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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        # type check for new data
        self._data = new_data

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets):
        # type check for new argets
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
                    self.targets.append(row)
            data = []
            data_dict = {filename: data for data, filename in self.data}
            for index, target in enumerate(self.targets):
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
                self.data.append((path, filename))
            elif self._strategy == "eager":
                image = self._read_data_file(path)
                self.data.append((image, filename))
        self._csv_to_labels()

    def load_data(self, root, strategy, format="csv", labels_path=None):
        # the user should be allowed to input a label path only if the format
        # is csv --> IMPLEMENT ERROR
        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if self._format == "csv":
            self._csv_load_data()

    def _get_item_helper(self, data, index):
        if bool(self._labels_path) or self._format == "hierarchical":
            return data, self.targets[index]
        else:
            return data

    def __getitem__(self, index: int):
        # different if data is loaded in eager way or lazy way
        if not bool(self.data):
            raise IndexError("No data available.")
        try:
            if self._strategy == "eager":
                data = self.data[index]
            elif self._strategy == "lazy":
                file_dir = self.data[index]
                data = self._read_data_file(file_dir)

            return self._get_item_helper(data, index)
        except IndexError:
            raise IndexError("Index out of range.")

    def __len__(self):
        # different if data is loaded in eager way or lazy way
        return len(self.data)

    def train_test_split(self, train_size, test_size, shuffle):
        # raise error: data hasn't been loaded yet
        data_len = len(self.data)
        train_len = int(data_len * train_size)

        train = copy.deepcopy(self)
        test = copy.deepcopy(self)

        if shuffle:
            if bool(self.targets):
                zipped_data = list(zip(self.data, self.targets))
                shuffled_data = random.sample(zipped_data, data_len)
                self.data, self.targets = zip(*shuffled_data)
            else:
                self.data = random.sample(self.data, data_len)

        train.data = self.data[:train_len]
        test.data = self.data[train_len::]

        if bool(self.targets):
            train.targets = self.targets[:train_len]
            test.targets = self.targets[train_len::]

        return train, test
