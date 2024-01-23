import os.path

import numpy as np
from PIL import Image


class ImageLoader:
    def _read_data_file(self, path):
        try:
            image_d = Image.open(path)
            image_d.load()
            image_d.show()
        except Exception as e:
            print(
                f"Error opening image {os.path.basename(path)}: {e}"
            )  # specify the error better
        #image_data = np.array(image)
        return image_d

    def __call__(self, path):
        return self._read_data_file(path)
