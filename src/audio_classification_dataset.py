from src.audio_loader import AudioLoader
from src.classification_dataset import ClassificationDataset


class AudioClassificationDataset(ClassificationDataset):
    def __init__(self) -> None:
        """
        Constructor of the class inheriting all the attributes from the
        super class
        """
        super().__init__()
        loader = AudioLoader()
        ClassificationDataset._read_data_file = loader
