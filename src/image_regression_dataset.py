from regression_dataset import RegressionDataset
from src.image_loader import ImageLoader


class ImageRegressionDataset(RegressionDataset):
    def __init__(self) -> None:
        """
        Constructor of the class inheriting all the attributes from the
        super class
        """
        super().__init__()
        loader = ImageLoader()
        ImageRegressionDataset._read_data_file = loader
