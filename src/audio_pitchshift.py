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
    preproces = PitchShifting(12)
    # path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio_classification_hierarchy/Cats/cat0548.wav"
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Cat/0a0da090aa9f0342444a7df4dc250c66.jpg"
    audio = librosa.load(path)
    new_audio = preproces(audio)
    print(new_audio)