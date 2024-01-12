import csv
import glob
import os
import os.path


class BaseDataset:
    def __init__(self):
        self.root = None
        self.format = None
        self.data = []
        self.labels = []

    def _read_data_file(self, filename):
        pass

    def _csv_to_labels(self):
        with open(self.labels_path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.labels.append(row)  # list or tuple?

    def _lazy_load_data(self, batch_size):
        if self.format == "csv":
            # batches + let user chose about the last batch
            data_list = os.listdir(self.root)
            if len(data_list) % batch_size != 0:
                # ask the user what to do about the last batchpass
            else: 
                for i in range(len(data_list) // batch_size):
                    self.data = data_list[i:i+batch_size]
                    self._eager_load_data()
                    i += batch_size   
                pass
        elif self.format == "hierarchical":
            # something for the last bath again
            pass

    def _eager_load_data(self):
        if self.format == "csv":
            for filename in os.listdir(self.root):
                self._read_data_file(filename)
        elif self.format == "hierarchical":
            # iterate over every class folder
            for class_dir in os.listdir(self.root):
                # get the class name from the directory name
                class_name = os.path.basename(class_dir)
                # iterate over every file
                for filename in os.listdir(class_name):
                    self._read_data_file(filename)
                    self.labels.append(
                        (filename, class_name)
                    )  # list or tuple?

    def load_data(self, root, strategy, format, labels_path=None):
        # the user should be allowed to input a label path only if the format is csv
        self.root = root
        self.strategy = strategy
        self.format = format
        self.labels_path = labels_path

        if strategy == "eager":
            self._eager_load_data()
        elif strategy == "lazy":
            self._lazy_load_data()

        if self.labels_path is not None:
            self._csv_to_labels()

    def __getitem__(self, index: int):
        if bool(self.data):
            if bool(self.labels):
                return (self.labels[index], self.data[index])
            else:
                return self.data[index]

    def __len__(self):
        return len(self.data)

    def train_test_split(self, train_size, test_size):
        pass
