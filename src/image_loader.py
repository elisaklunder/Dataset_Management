import os.path

import numpy as np
from PIL import Image


class ImageLoader:
    def _read_data_file(self, path, filename):
        try:
            image = Image.open(os.path.join(path, filename))
        except Exception as e:
            print(
                f"Error opening image {filename}: {e}"
            )  # specify the error better
        image_data = np.array(image)
        return image_data

    def __call__(self, path, filename):
        return self._read_data_file(path, filename)
