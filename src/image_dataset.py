import os.path

import numpy as np
from PIL import Image

from src.base_dataset import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def _read_data_file(self, path, filename):
        image = Image.open(os.path.join(path, filename))
        # maybe open/load is the solution to lazy/eager?
        image_data = np.array(image)
        return image_data, filename
