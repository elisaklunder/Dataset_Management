import os.path

import numpy as np
from PIL import Image

from src.base_dataset import BaseDataset


class ImageDataset(BaseDataset):

    def __init__(self):
        super().__init__()

    def _read_data_file(self, path, filename):
        try:
            image = Image.open(os.path.join(path, filename))
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
        image_data = np.array(image)
        return image_data, filename
