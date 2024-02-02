import os

from base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    # implements hierarchical strategy
    def __init__(self) -> None:
        super().__init__()

    def load_data(
        self,
        root: str,
        strategy: str,
        format: str = "csv",
        labels_path: str = None,
    ) -> None:
        super().load_data(root, strategy, format, labels_path)
        if self._format == "hierarchical":
            self._hierarchical_load_data()

    def _hierarchical_load_data(self) -> None:
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
