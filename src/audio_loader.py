import os.path

import librosa
import numpy as np

#from src.base_dataset import BaseDataset


class AudioLoader:
    def _read_data_file(self, path):
        audio_tuple = librosa.load(path) #y and sampling rate --> it can be npacked
        return audio_tuple

    def __call__(self, path):
        return self._read_data_file(path)


def main():
    path = '/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio/Cats/cat0001.wav'
    audio = AudioLoader()
    y, sr = audio._read_data_file(path)
    print(y)
    print(sr)


if __name__ == "__main__":
    main()
