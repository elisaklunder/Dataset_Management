import csv
import os
import os.path
import random


class BaseDataset:
    def __init__(self):
        self._root = None
        self._format = None
        self._data = []

    def _read_data_file(self, filename):
        pass

    def _csv_to_labels(self):
        labels = []
        file_list = os.listdir(self._root)
        csv_path = None
        for file in file_list:
            if file.endswith(".csv"):
                csv_path = os.path.join(self._root, file)
                break
        if csv_path is not None:
            with open(csv_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    labels.append(row)  # list of tuples?
            return labels
        else:
            #error path not correct or no csv file found
            pass

    def _lazy_load_data(self):
        if self._format == "csv":
            # Store filenames for lazy loading
            self._data = [filename for filename in os.listdir(self._root)]
        elif self._format == "hierarchical":
            # Store class directories and filenames for lazy loading
            self._data = [
                (class_dir, filename)
                for class_dir in os.listdir(self._root)
                for filename in os.listdir(class_dir)
            ]

    """def _lazy_load_data_maybe(self, batch_size):
        if self._format == "csv":
            # batches + let user chose about the last batch --> this is
            # actually not the case --> it only means that the data is
            # uploaded when is actually used.
            data_list = os.listdir(self._root)
            if len(data_list) % batch_size != 0:
                # ask the user what to do about the last batchpass
                pass
            else:
                for i in range(len(data_list) // batch_size):
                    self._data = data_list[i : i + batch_size]
                    self.__eager_load_data()
                    i += batch_size
                pass
        elif self._format == "hierarchical":
            # something for the last bath again
            pass"""

    def _eager_load_data(self):
        if self._format == "csv":
            for filename in os.listdir(self._root):
                self._read_data_file(filename)
        elif self._format == "hierarchical":
            # iterate over every class folder
            for class_dir in os.listdir(self._root):
                # get the class name from the directory name
                class_name = os.path.basename(class_dir)
                # iterate over every file
                for filename in os.listdir(class_name):
                    self._read_data_file(filename)
                    self._labels.append(
                        (filename, class_name)
                    )  # list or tuple?

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

        if self._labels_path is not None:
            labels = self._csv_to_labels()
            labels_dict = {item[1]: item[2] for item in labels}
            self._data = [
                (image, labels_dict[filename])
                for image, filename in self._data
            ]

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
