import os.path

import numpy as np
import PIL
from PIL import Image

from abc_loader import DataLoaderABC


class ImageLoader(DataLoaderABC):
    def _read_data_file(self, path: str) -> np.ndarray:
        try:
            image = Image.open(path)
        except PIL.Image.UnidentifiedImageError:
            raise PIL.Image.UnidentifiedImageError(
                f"Error trying to open file '{os.path.basename(path)}'. The \
specified file could be of an incorrect format."
            )
        image_data = np.array(image)
        return image_data
