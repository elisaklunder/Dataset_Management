import os.path

import numpy as np
import PIL
from PIL import Image

from abc_loader import DataLoaderABC


class ImageLoader(DataLoaderABC):
    def _read_data_file(self, path: str) -> np.ndarray:
        """
        Given the directory of an an file, the method should be able
        to load the data into the program

        Args:
            path (str): directory to an audio file

        Raises:
            PIL.Image.UnidentifiedImageError: This is an error from librosa.
            The program is just re-raising it, whenever the file cannot be
            opened or is in the wrong format

        Returns:
            np.ndarray: array representation of the image
        """
        try:
            image = Image.open(path)
        except PIL.Image.UnidentifiedImageError:
            raise PIL.Image.UnidentifiedImageError(
                f"Error trying to open file '{os.path.basename(path)}'. The \
specified file could be of an incorrect format."
            )
        image_data = np.array(image)
        return image_data
