from base_dataset import BaseDataset
import os


class ClassificationDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    # implement hierarchical

    def _lazy_load_data(self):
        if self._format == "csv":
            super()._lazy_load_data()

        elif self._format == "hierarchical":
            print("in the lazy loader")
            # Store class directories and filenames for lazy loading
            self._data = [
                (os.path.join(self._root, filename), class_dir)
                for class_dir in os.listdir(self._root)
                for filename in os.listdir(os.path.join(self._root, class_dir))
            ]


    def _eager_load_data(self):
        if self._format == "csv":
            super()._eager_load_data()
    
        elif self._format == "hierarchical":
            # iterate over every class folder
            for class_name in os.listdir(self._root):
                class_dir = os.path.join(self._root, class_name)
                # iterate over every file
                for filename in os.listdir(class_dir):
                    image, _ = self._read_data_file(class_dir, filename)
                    self._data.append((image, class_name))

    pass
