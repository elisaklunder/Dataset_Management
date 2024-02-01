import librosa

from errors import Errors
from preprocessing_ABC import PreprocessingTechniqueABC


class Resampling(PreprocessingTechniqueABC):
    def __init__(self, new_sampling_rate: int) -> None:
        error = Errors()
        error.type_check("new_sampling_rate", new_sampling_rate, int)
        error.ispositive("new_sampling_rate", new_sampling_rate)
        self._new_sampling_rate = new_sampling_rate

    def _preprocessing_logic(self, audio: tuple) -> tuple:
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


if __name__ == "__main__":
    preproces = Resampling(875)
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio_classification_hierarchy/Cats/cat0548.wav"
    audio = librosa.load(path)
    new_audio = preproces(audio)
