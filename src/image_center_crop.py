import numpy as np
from PIL import Image

from preprocessing_ABC import PreprocessingTechniqueABC


class ImageCenterCrop(PreprocessingTechniqueABC):
    def __init__(self, width, height):
        self._width = width
        self.height = height

    def __call__(self, dataset):
        return self._dataset_processing(dataset)

    def _crop_image(self, image):
        H, W = np.shape(image)
        H_center = H // 2
        W_center = W // 2
        top = max(H_center - self._height // 2, 0)
        bottom = min(H_center + self._height // 2, H)
        left = max(W_center - self._width // 2, 0)
        right = min(W_center + self._width // 2, W)
        cropped_image = image[top:bottom, left:right]
        return cropped_image

    def _dataset_processing(self, dataset):
        cropped_data = []
        for image in dataset.data:
            cropped_image = self._crop_image(image)
            cropped_data.append(cropped_image)
        if bool(dataset.targets):
            return cropped_data, dataset.targets
        else:
            return cropped_data
