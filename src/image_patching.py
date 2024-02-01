import random

import numpy as np
from PIL import Image

from preprocessing_ABC import PreprocessingTechniqueABC


class ImagePatching(PreprocessingTechniqueABC):
    def __init__(self, color, height, width):
        # implement error if the box is bigger than the image
        # handle everything  when crop is too big
        self._color = color
        self._width = width
        self._height = height

    def _preprocessing_logic(self, array):
        image = Image.fromarray(array)
        width, height = image.size
        x_left = random.randint(0, width - self._width)
        y_left = random.randint(0, height - self._height)

        x_right = x_left + self._width
        y_right = y_left + self._height
        box = (x_left, y_left, x_right, y_right)
        region = Image.new("RGB", (self._width, self._height), self._color)

        image.paste(region, box)
        image_array = np.array(image)

        return image_array
