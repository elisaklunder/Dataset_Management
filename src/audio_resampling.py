import librosa

from errors import Errors
from preprocessing_ABC import PreprocessingTechniqueABC


class Resampling(PreprocessingTechniqueABC):
    def __init__(self, new_sampling_rate: int) -> None:
        """
        Constructor of the class

        Args:
            new_sampling_rate (int): integer value indicating what the new
            sampling rate of the signal will be
        """
        error = Errors()
        error.type_check("new_sampling_rate", new_sampling_rate, int)
        error.ispositive("new_sampling_rate", new_sampling_rate)
        self._new_sampling_rate = new_sampling_rate

    def _preprocessing_logic(self, audio: tuple) -> tuple:
        """
        Given a tuple, containing samples and sampling rate, the method
        re-samples the audio based on sampling frequency specified by the user

        Args:
            audio (tuple): audio data to be preprocessed. The values inside
            the tuple are the samples and the sampling rates respectively.

        Raises:
            ValueError: This is an error from librosa. The program is
            just re-raising it, whenever the data is in the wrong format

        Returns:
            tuple: audio data with the new samples and the relative sampling
            frequency
        """
        samples, sampling_rate = audio
        try:
            resampled_signal = librosa.resample(
                samples,
                orig_sr=sampling_rate,
                target_sr=self._new_sampling_rate,
            )
        except librosa.util.exceptions.ParameterError:
            raise ValueError("Audio data is in the wrong format")
        return resampled_signal, self._new_sampling_rate
