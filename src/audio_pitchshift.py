import librosa

from errors import Errors
from preprocessing_ABC import PreprocessingTechniqueABC


class PitchShifting(PreprocessingTechniqueABC):
    def __init__(self, pitch_shift: int) -> None:
        self._error = Errors()
        self._error.type_check("pitch_shift", pitch_shift, int)
        self._error.ispositive("pitch_shift", pitch_shift)
        self._pitch_shift = pitch_shift

    def _preprocessing_logic(self, audio: tuple) -> tuple:
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


if __name__ == "__main__":
    preproces = PitchShifting(45)
    # path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio_classification_hierarchy/Cats/cat0548.wav"
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Dog/0a3d571cec18903dd963639dddf5587d.jpg"
    audio = librosa.load(path)
    new_audio = preproces(audio)
    print(len(new_audio[0]), len(audio[0]))
