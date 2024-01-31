from PIL import Image
from preprocessing_ABC import PreprocessingTechniqueABC
import random
import matplotlib
import numpy as np


class ImagePatching(PreprocessingTechniqueABC):
    def __init__(self, color, height, width):
        # implement error if the box is bigger than the image
        #handle everything  when crop is too big
        self._color = color
        self._width = width
        self._height = height

    def _patching(self, array):
        
        image = Image.fromarray(array)
        width, height = image.size
        x_left = random.randint(0, width - self._width)
        y_left = random.randint(0, height - self._height)

        x_right = x_left + self._width
        y_right = y_left + self._height
        box = (x_left, y_left, x_right, y_right)
        # box_image = image.crop(box)
        region = Image.new("RGB", (self._width, self._height), self._color)
        
        image.paste(region, box)
        image_array = np.array(image)

        return image_array

    def __call__(self, image):
        return self._patching(image)


if __name__ == "__main__":
    preproces = ImagePatching("yellow", 155, 145)
    path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/image_classification_hierarchy/Cat/0a0da090aa9f0342444a7df4dc250c66.jpg"
    image = Image.open(path)
    new_image = np.array(image)
    preproces(new_image)

