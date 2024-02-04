from base_dataset import BaseDataset


class RegressionDataset(BaseDataset):
    """
    Class implementing regression datasets.
    """

    def __init__(self) -> None:
        super().__init__()

    def load_data(
        self,
        root: str,
        strategy: str,
        format: str,
        labels_path: str = None,
    ):
        if format == "hierarchical":
            raise ValueError(
                "A regression dataset can't be organized in \
a hierarchical folder structure"
            )
        super().load_data(root, strategy, format, labels_path)
