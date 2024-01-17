from src.image_loader import ImageLoader
from regression_dataset import RegressionDataset


class ImageRegressionDataset(RegressionDataset):
    loader = ImageLoader()
    RegressionDataset._read_data_file = loader
