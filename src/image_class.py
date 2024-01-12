from base_class import BaseDataset
import numpy as np
from PIL import Image
import os.path


class ImageDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def _read_data_file(self, filename):
        image = Image.open(os.path.join(self.root, filename))
        image_data = np.array(image)
        self.data.append(image_data)