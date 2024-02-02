import random

import numpy as np
from PIL import Image

from errors import Errors
from preprocessing_ABC import PreprocessingTechniqueABC


class ImagePatching(PreprocessingTechniqueABC):
    def __init__(self, color: str, height: int, width: int) -> None:
        """
        Constructor of the class.

        Args:
            color (str): string indicating color of the patch
            height (int): integer showing the height of the patch
            width (int): integer showing the width of the patch
        """
        self._errors = Errors()
        self._errors.type_check("color", color, str)
        self._errors.type_check("width", width, int)
        self._errors.type_check("height", height, int)
        self._errors.ispositive("width", width)
        self._errors.ispositive("height", height)
        self._color = color
        self._width = width
        self._height = height

    def _preprocessing_logic(self, array: np.ndarray) -> np.ndarray:
        """
        Method implementing the random patching of an image, given the size,
        width and color of the patch.

        Args:
            array (np.ndarray): array containing the data of the image to be
            patched

        Raises:
            ValueError: if the image is smaller than the specified patch size.

        Returns:
            np.ndarray: image in array format with the random patch on it.
        """
        image = Image.fromarray(array)
        width, height = image.size
        if width < self._width or height < self._height:
            raise ValueError(
                "Image can't be smaller than the specified patch size."
            )
        x_left = random.randint(0, width - self._width)
        y_left = random.randint(0, height - self._height)
        x_right = x_left + self._width
        y_right = y_left + self._height
        box = (x_left, y_left, x_right, y_right)
        region = Image.new("RGB", (self._width, self._height), self._color)
        image.paste(region, box)
        image_array = np.array(image)
        return image_array
