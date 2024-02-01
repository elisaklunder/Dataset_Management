from src.classification_dataset import ClassificationDataset
from src.image_loader import ImageLoader


class ImageClassificationDataset(ClassificationDataset):
    def __init__(self) -> None:
        super().__init__()
        loader = ImageLoader()
        ClassificationDataset._read_data_file = loader
