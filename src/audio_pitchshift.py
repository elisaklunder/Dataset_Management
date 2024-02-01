import librosa
from preprocessing_ABC import PreprocessingTechniqueABC


class PitchShifting(PreprocessingTechniqueABC):
    def __init__(self, pitch_shift):
        self._pitch_shift = pitch_shift

    def _preprocessing_logic(self, audio):
        sample, sampling_rate = audio
        shifted_audio = librosa.effects.pitch_shift(
            sample, sr=sampling_rate, n_steps=self._pitch_shift,bins_per_octave=self._pitch_shift
            )
        return shifted_audio, sampling_rate
    

if __name__ == "__main__":
    preproces = PitchShifting(0)
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio_classification_hierarchy/Cats/cat0548.wav"
    audio = librosa.load(path)
    new_audio = preproces(audio)
    print(len(new_audio[0]), len(audio[0]))
