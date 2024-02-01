import numpy as np

from preprocessing_ABC import PreprocessingTechniqueABC


class ImageCenterCrop(PreprocessingTechniqueABC):
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def _preprocessing_logic(self, data):
        H, W, _ = np.shape(data)
        H_center = H // 2
        W_center = W // 2
        top = max(H_center - self._height // 2, 0)
        bottom = min(H_center + self._height // 2, H)
        left = max(W_center - self._width // 2, 0)
        right = min(W_center + self._width // 2, W)
        cropped_image = data[top:bottom, left:right]
        return cropped_image
