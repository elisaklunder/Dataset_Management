import csv
import os
import os.path
import random
import copy


class BaseDataset:
    def __init__(self):
        self._root = None
        self._labels_path = None
        self._format = None
        self._data = []
        self._strategy = None
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data

    def _csv_to_labels(self):
        if self._labels_path is not None:
            labels = []
            with open(self._labels_path, "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    labels.append(row)
            labels_dict = {item[1]: item[2] for item in labels}
            self.data = [
                (image, labels_dict[filename])
                for image, filename in self.data
            ]
        else:
            self.data = [image for image, _ in self.data]

    def _lazy_load_data(self):
        self.data = [
            (os.path.join(self._root, filename), filename)
            for filename in os.listdir(self._root)
        ]
        self._csv_to_labels()

    def _eager_load_data(self):
        for filename in os.listdir(self._root):
            image, filename = self._read_data_file(self._root, filename)
            self.data.append((image, filename))
        self._csv_to_labels()

    def load_data(self, root, strategy, format="csv", labels_path=None):
        # the user should be allowed to input a label path only if the format
        # is csv
        self._root = root
        self._strategy = strategy
        self._format = format
        self._labels_path = labels_path

        if self._strategy == "eager":
            self._eager_load_data()
        elif self._strategy == "lazy":
            self._lazy_load_data()

    def __getitem__(self, index: int):
        # different if data is loaded in eager way or lazy way
        if not bool(self.data):
            # raise error no data
            pass
        try:
            if self._strategy == "eager":
                return self.data[index]
            elif self._strategy == "lazy":
                if bool(self._labels_path):
                    file_dir, target = self.data[index]
                    data, _ = self._read_data_file(self._root, file_dir)
                    return data, target
                elif self._format == "hierarchical":
                    file_dir, target = self.data[index]
                    root_path = os.path.join(self._root, target)
                    file_name = os.path.basename(file_dir)
                    data, _ = self._read_data_file(root_path, file_name)
                    return data, target
                else:
                    file_dir = self.data[index]
                    data, _ = self._read_data_file(self._root, file_dir)
                    return data
        except:
            # raise error data out of index
            pass

    def __len__(self):
        # different if data is loaded in eager way or lazy way
        return len(self.data)

    def train_test_split(self, train_size, test_size, shuffle):
        # raise error: data hasn't been loaded yet
        data_len = len(self.data)
        train_len = int(data_len * train_size)
        test_len = int(data_len * test_size)

        if shuffle:
            shuffled_data = random.sample(self.data, data_len)
            train_data = shuffled_data[:train_len]
            test_data = shuffled_data[train_len : train_len + test_len]
        else:
            train_data = self.data[:train_len]
            test_data = self.data[train_len : train_len + test_len]
        train = copy.deepcopy(self)
        train.data = train_data
        test = copy.deepcopy(self)
        test.data = test_data
        return train, test
