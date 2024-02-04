from src.audio_loader import AudioLoader
from regression_dataset import RegressionDataset


class AudioRegressionDataset(RegressionDataset):
    def __init__(self) -> None:
        """
            Class to be further implemented if the user wants to carry out
            actual computations once the data was loaded
        """
        super().__init__()
        loader = AudioLoader()
        AudioRegressionDataset._read_data_file = loader
