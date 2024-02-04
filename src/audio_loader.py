import os.path

import librosa
from audioread.exceptions import NoBackendError

from abc_loader import DataLoaderABC


class AudioLoader(DataLoaderABC):
    def _read_data_file(self, path: str) -> tuple:
        """
        Given the directory of an audio file, the method should be able
        to load the data into the program

        Args:
            path (str): directory to an audio file

        Raises:
            NoBackendError: This is an error from librosa. The program is
            re-raising it whenever the file cannot be opened or is
            in the wrong format

        Returns:
            tuple: tuple containing the samples of the audio files and the
            sampling rate
        """
        try:
            audio_tuple = librosa.load(path)
        except NoBackendError:
            raise NoBackendError(
                f"Error trying to open file '{os.path.basename(path)}'. The \
specified file could be of an incorrect format."
            )
        return audio_tuple
