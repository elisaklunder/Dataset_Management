import os

from base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    # implement hierarchical

    def _lazy_load_data(self):
        if self._format == "csv":
            super()._lazy_load_data()

        elif self._format == "hierarchical":
            # Store class directories and filenames for lazy loading
            for class_name in os.listdir(self._root):
                for filename in os.listdir(
                    os.path.join(self._root, class_name)
                ):
                    class_path = os.path.join(self._root, class_name)
                    self.data.append(os.path.join(class_path, filename))
                    self.targets.append(class_name)

    def _eager_load_data(self):
        if self._format == "csv":
            super()._eager_load_data()

        elif self._format == "hierarchical":
            # iterate over every class folder
            for class_name in os.listdir(self._root):
                class_dir = os.path.join(self._root, class_name)
                # iterate over every file
                for filename in os.listdir(class_dir):
                    path = os.path.join(class_dir, filename)
                    data = self._read_data_file(path)
                    self.data.append(data)
                    self.targets.append(class_name)
