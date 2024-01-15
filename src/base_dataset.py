import csv
import os
import os.path
import random


class BaseDataset:
    def __init__(self):
        self._root = None
        self._labels_path = None
        self._format = None
        self._data = []

    def _read_data_file(self, filename):
        pass

    def _csv_to_labels(self):
        if self._labels_path is not None:
            labels = []
            with open(self._labels_path, "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    labels.append(row)  # list of tuples?
            labels_dict = {item[1]: item[2] for item in labels}
            self._data = [
                (image, labels_dict[filename])
                for image, filename in self._data
            ]
        else:
            self._data = [image for image, _ in self._data]

    def _lazy_load_data(self):
        if self._format == "csv":
            # Store filenames for lazy loading
            self._data = [
                (filename, filename) for filename in os.listdir(self._root)
            ]
            self._csv_to_labels()
        elif self._format == "hierarchical":
            # Store class directories and filenames for lazy loading
            self._data = [
                (filename, class_dir)
                for class_dir in os.listdir(self._root)
                for filename in os.listdir(os.path.join(self._root, class_dir))
            ]

    def _eager_load_data(self):
        if self._format == "csv":
            for filename in os.listdir(self._root):
                image, filename = self._read_data_file(self._root, filename)
                self._data.append((image, filename))
            self._csv_to_labels()

        elif self._format == "hierarchical":
            # iterate over every class folder
            for class_name in os.listdir(self._root):
                class_dir = os.path.join(self._root, class_name)
                # iterate over every file
                for filename in os.listdir(class_dir):
                    image, _ = self._read_data_file(class_dir, filename)
                    self._data.append((image, class_name))

    def load_data(self, root, strategy, format, labels_path=None):
        # the user should be allowed to input a label path only if the format
        # is csv
        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if strategy == "eager":
            self._eager_load_data()
        elif strategy == "lazy":
            self._lazy_load_data()

    def __getitem__(self, index: int):
        # different if data is loaded in eager way or lazy way
        if not bool(self._data):
            # raise error no data
            pass
        try:
            return self._data[index]
        except:
            # raise error data out of index
            pass

    def __len__(self):
        # different if data is loaded in eager way or lazy way
        return len(self._data)

    def train_test_split(self, train_size, test_size, shuffle):
        # raise error: data hasn't been loaded yet
        data_len = len(self._data)
        train_len = int(data_len * train_size)
        test_len = int(data_len * test_size)

        if shuffle:
            shuffled_data = random.sample(self._data, data_len)
            train_data = shuffled_data[:train_len]
            test_data = shuffled_data[train_len : train_len + test_len]
        else:
            train_data = self._data[:train_len]
            test_data = self._data[train_len : train_len + test_len]

        return train_data, test_data
