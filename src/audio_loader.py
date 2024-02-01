import os.path

import librosa
from audioread.exceptions import NoBackendError
import numpy as np
from abc_loader import DataLoaderABC


class AudioLoader(DataLoaderABC):
    def _read_data_file(self, path):
        try:
            audio_tuple = librosa.load(path) 
        except NoBackendError:
            raise NoBackendError(f"Error trying to open path '{path}'. The \
specified file could be of an incorrect format.")
        # y and sampling rate --> it can be unpacked
        return audio_tuple



def main():
    #path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio/Cats/cat0001.wav"
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Cat/0a0da090aa9f0342444a7df4dc250c66.jpg"
    audio = AudioLoader()
    y, sr = audio._read_data_file(path)
    print(y)
    print(sr)


if __name__ == "__main__":
    main()
