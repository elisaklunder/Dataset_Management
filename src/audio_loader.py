import os.path

import librosa
import numpy as np

from src.base_dataset import BaseDataset


class AudioLoader:
    def _read_data_file(self, path, filename):
        audio_tuple = librosa.load(filename)
        return audio_tuple

    def __call__(self, path, filename):
        return self._read_data_file(path, filename)
