from preprocessing_ABC import PreprocessingTechniqueABC
import librosa
from errors import Errors


class Resampling(PreprocessingTechniqueABC):
    def __init__(self, new_sampling_rate):
        error = Errors()
        error.type_check("new_sampling_rate", new_sampling_rate, int)
        error.ispositive("new_sampling_rate", new_sampling_rate)
        self._new_sampling_rate = new_sampling_rate

    def _preprocessing_logic(self, audio):
        samples, sampling_rate = audio
        resampled_signal = librosa.resample(samples, orig_sr=sampling_rate, target_sr=self._new_sampling_rate)
        return resampled_signal, self._new_sampling_rate


if __name__ == "__main__":
    preproces = Resampling(875)
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio_classification_hierarchy/Cats/cat0548.wav"
    audio = librosa.load(path)
    print(audio)
    new_audio = preproces(audio)
    print(new_audio)
