# import librosa

from errors import Errors
from preprocessing_ABC import PreprocessingTechniqueABC


class PitchShifting(PreprocessingTechniqueABC):
    def __init__(self, pitch_shift: int) -> None:
        """
        Constructor of the class

        Args:
            pitch_shift (int): integer value indicating the value of the
            pitch shifting
        """
        self._error = Errors()
        self._error.type_check("pitch_shift", pitch_shift, int)
        self._error.ispositive("pitch_shift", pitch_shift)
        self._pitch_shift = pitch_shift

    def _preprocessing_logic(self, audio: tuple) -> tuple:
        """
        Method implementing the pitch shift for a given audio

        Args:
            audio (tuple): tuple containing the samples and the sampling rate

        Raises:
            ValueError: This is an error from librosa. The program is
            just re-raising it, whenever the file is
            in the wrong format

        Returns:
            tuple: tuple containing the samples and the sampling rate of the
            new signal shifted pitch shifted.
        """
        sample, sampling_rate = audio
        try:
            shifted_audio = librosa.effects.pitch_shift(
                sample,
                sr=sampling_rate,
                n_steps=self._pitch_shift,
                bins_per_octave=self._pitch_shift,
            )
        except librosa.util.exceptions.ParameterError:
            raise ValueError("Audio data is in the wrong format")
        return shifted_audio, sampling_rate
