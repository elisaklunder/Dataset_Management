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
        """
        Public method used to load the data. It inherits the logic from the
        load_data method in BaseDataset, but implements a check before that to
        rule out the possibility that a user will mistakenly indicate a
        hierarchical folder structure for their data when this is not possible
        in datasets made for regression

        Args:
            root (str): directory indicating a path to the data to be loaded.

            strategy (str): string specifying whether the data is loaded in a
            lazy or eager fashion.

            format (str, optional): string indicating the structure of the
            data. Defaults to "csv".

            labels_path (str, optional): directory indicating the path to the
            labels, if any. Defaults to None.


        Raises:
            ValueError: if the specified format is hierarchical

        """
        if format == "hierarchical":
            raise ValueError(
                "A regression dataset can't be organized in \
a hierarchical folder structure"
            )
        super().load_data(root, strategy, format, labels_path)
