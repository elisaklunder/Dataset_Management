import os.path

import librosa
import numpy as np
from audioread import NoBackendError


class AudioLoader:
    def _read_data_file(self, path):
        audio_tuple = librosa.load(path)  # y and sampling rate --> it can be unpacked
        
        return audio_tuple

    def __call__(self, path):
        return self._read_data_file(path)


def main():
    #path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio/Cats/cat0001.wav"
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Cat/0a0da090aa9f0342444a7df4dc250c66.jpg"
    audio = AudioLoader()
    y, sr = audio._read_data_file(path)
    print(y)
    print(sr)


if __name__ == "__main__":
    main()
