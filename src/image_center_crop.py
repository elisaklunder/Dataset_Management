import numpy as np
from PIL import Image

from preprocessing_ABC import PreprocessingTechniqueABC


class ImageCenterCrop(PreprocessingTechniqueABC):
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def _preprocessing_logic(self, data):
        H, W, _ = np.shape(image)
        H_center = H // 2
        W_center = W // 2
        top = max(H_center - self._height // 2, 0)
        bottom = min(H_center + self._height // 2, H)
        left = max(W_center - self._width // 2, 0)
        right = min(W_center + self._width // 2, W)
        cropped_image = image[top:bottom, left:right]
        return cropped_image




if __name__ == "__main__":
    crop = ImageCenterCrop(100, 100)
    path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\image_regression_csv\images_poly\000cf421-6725-4dee-bf37-04525ba04340.png"
    image = Image.open(path)
    image.show()
    image = np.array(image)
    cropped_image = crop(image)
    showoff = Image.fromarray(cropped_image)
    showoff.show()
