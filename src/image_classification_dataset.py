from src.classification_dataset import ClassificationDataset
from src.image_loader import ImageLoader


class ImageClassificationDataset(ClassificationDataset):
    def __init__(self) -> None:
        """
        Constructor of the class inheriting all the attributes from the
        super class
        """
        super().__init__()
        loader = ImageLoader()
        ImageClassificationDataset._read_data_file = loader
