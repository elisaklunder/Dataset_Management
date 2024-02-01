from src.audio_loader import AudioLoader
from regression_dataset import RegressionDataset


class AudioRegressionDataset(RegressionDataset):
    """
    Class to be further implemented if the user wants to carry out
    actual computations once the data was loaded
    """
    loader = AudioLoader()
    RegressionDataset._read_data_file = loader
