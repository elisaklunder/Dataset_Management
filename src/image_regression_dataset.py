from regression_dataset import RegressionDataset
from src.image_loader import ImageLoader


class ImageRegressionDataset(RegressionDataset):
    loader = ImageLoader()
    RegressionDataset._read_data_file = loader
