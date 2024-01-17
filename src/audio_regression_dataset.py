from src.audio_loader import AudioLoader
from regression_dataset import RegressionDataset


class AudioRegressionDataset(RegressionDataset):
    loader = AudioLoader()
    RegressionDataset._read_data_file = loader
