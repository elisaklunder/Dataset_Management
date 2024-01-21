from PIL import Image
from preprocessing_ABC import PreprocessingTechniqueABC
import random
import matplotlib


class ImagePatching(PreprocessingTechniqueABC):
    def __init__(self, color, height, width):
        self._color = color
        self._width = width
        self._height = height

    def _patching(self, image):
        width, height = image.size(image)
        x_left = random.randint(0, self._width - width)
        y_left = random.random(0, self.heigh - height)

        x_right = x_left + width
        y_right = y_left - height

        box = (x_left, y_left, x_right, y_right)
        region = image.crop(box)
        region = Image.new("RGB", (self._width, self._height), self._color)
        image.paste(region, box)

        return image

    def _patchingArray(self, image):
        width, height = image.size(image)
        x_left = random.randint(0, self._width - width)
        y_left = random.random(0, self.heigh - height)
        x_right = x_left + width

        rgb = matplotlib.colors.to_rgb(self._color)

        for i in range(height):
            image[y_left + i, x_left:x_right] = rgb = rgb

        return image

    def __call__(self, image):
        return self._patching(image)
