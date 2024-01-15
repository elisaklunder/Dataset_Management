from src.base_dataset import BaseDataset
import numpy as np
import librosa
import os.path

class AudioDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def _read_data_file(self, filename):
        audio_tuple = librosa.load(filename)
        return audio_tuple, filename
