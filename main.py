import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.getcwd() + "/src/")
# from src.audio_classification_dataset import AudioClassificationDataset
from src.batchloader import BatchLoader
from src.image_center_crop import ImageCenterCrop
from src.image_classification_dataset import ImageClassificationDataset
from src.image_patching import ImagePatching
from src.preprocessing_pipeline import PreprocessingPipeline


def main():
    image = ImageClassificationDataset()
    # audio = AudioClassificationDataset()

    # JULIA
    # root_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_hierarchy"

    # root_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_csv/images_poly"

    # labels_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_csv/poly_targets_regression.csv"
    # root_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/audio"

    # ELI
    root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\image_regression_csv\images_poly"
    # root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\image_classification_hierarchy"
    labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\image_regression_csv\poly_targets_regression.csv"
    image.load_data(root_path, "lazy", "csv")
    print(image[3])
    train, test = image.train_test_split(
        train_size=0.6, shuffle=True
    )


    # TESTING BATCHLOADER FUNCTIONALITY

    # batcher = BatchLoader()

    # batcher.create_batches(train, 51, "shuffle", False)
    # print(len(batcher))

    # show original image
    # path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\image_regression_csv\images_poly\000cf421-6725-4dee-bf37-04525ba04340.png"
    # image = Image.open(path)
    # image.show()

    # define pipeline
    # crop = ImageCenterCrop(100, 170)
    # patch = ImagePatching("yellow", 15, 15)
    # pipeline = PreprocessingPipeline(crop, patch)

    # apply pipeline
    # image = np.array(image)
    # data = (image, 3)
    # preprocessed_image = pipeline(data)
    # preprocessed_image = Image.fromarray(preprocessed_image)
    # preprocessed_image.show()


if __name__ == "__main__":
    main()
