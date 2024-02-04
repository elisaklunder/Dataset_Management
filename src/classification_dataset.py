import os

from base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    def __init__(self) -> None:
        """
        Constructor of the class.
        """
        super().__init__()

    def load_data(
        self,
        root: str,
        strategy: str,
        format: str,
        labels_path: str = None,
    ) -> None:
        """
        Public method, overriding the method in the Basedataset. It implements
        the possibility to load data when the files are in hierarchical
        folders, while also keeping everything that was implemented in the
        parent class.

        Args:
            root (str): root (str): directory indicating a path to the data to
            be loaded.

            strategy (str): string specifying whether the data is loaded in a
            lazy or eager fashion.

            format (str, optional): string indicating the structure of the
            data. Defaults to "csv".

            labels_path (str, optional): directory indicating the path to the
            labels, if any. Defaults to None.

        """
        super().load_data(root, strategy, format, labels_path)
        if self._format == "hierarchical":
            self._hierarchical_load_data()

    def _hierarchical_load_data(self) -> None:
        """
        Private method that implements the loading of the data when the files
        are organized in hierarchical subfolders
        """
        temp_data = []
        temp_targets = []
        for class_name in os.listdir(self._root):
            if class_name != ".DS_Store":
                class_dir = os.path.join(self._root, class_name)
                for filename in os.listdir(class_dir):
                    if filename == ".DS_Store":
                        continue
                    path = os.path.join(class_dir, filename)
                    if self._strategy == "lazy":
                        temp_data.append(path)
                    if self._strategy == "eager":
                        data_array = self._read_data_file(path)
                        temp_data.append(data_array)
                    temp_targets.append(class_name)
        self.data = temp_data
        self.targets = temp_targets
