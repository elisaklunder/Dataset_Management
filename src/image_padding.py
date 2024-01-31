from PIL import Image
from preprocessing_ABC import PreprocessingTechniqueABC
import random
import matplotlib
import numpy as np


class ImagePatching(PreprocessingTechniqueABC):
    def __init__(self, color, height, width):
        self._color = color
        self._width = width
        self._height = height

    def _patching(self, array):
        
        image = Image.fromarray(array)
        width, height = image.size
        x_left = random.randint(0, width - self._width)
        y_left = random.randint(0, height - self._height)

        x_right = x_left + width
        y_right = y_left + height
        box = (x_left, y_left, x_right, y_right)
        region = Image.new("RGB", (self._width, self._height), self._color)
        image.paste(region, box)
        image.show()
        image_array = np.array(image)

        return image_array

    #this works if the image is kept as an array
    def _patchingArray(self, image):
        width, height = image.size(image)
        x_left = random.randint(0, self._width - width)
        y_left = random.random(0, self.heigh - height)
        x_right = x_left + width

        rgb = matplotlib.colors.to_rgb(self._color)

        for i in range(height):
            image[y_left + i, x_left:x_right] = rgb

        return image

    def __call__(self, image):
        return self._patching(image)


if __name__ == "__main__":
    preproces = ImagePatching("blue", 14, 14)
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Cat/0a0da090aa9f0342444a7df4dc250c66.jpg"
    image = Image.open(path)
    new_image = np.array(image)
    preproces(new_image)

