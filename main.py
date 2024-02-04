import os
import sys

import sounddevice as sd
from PIL import Image

sys.path.append(os.getcwd() + "/src/")
from src.audio_classification_dataset import AudioClassificationDataset
from src.audio_pitchshift import PitchShifting
from src.audio_regression_dataset import AudioRegressionDataset
from src.audio_resampling import Resampling
from src.batchloader import BatchLoader
from src.image_center_crop import ImageCenterCrop
from src.image_classification_dataset import ImageClassificationDataset
from src.image_patching import ImagePatching
from src.image_regression_dataset import ImageRegressionDataset
from src.preprocessing_pipeline import PreprocessingPipeline


def main():
    # # DATASETS LOADING # #
    # In this first section we showcase the loading of data using our four
    # classes (ImageRegression, ImageClassification, AudioRegression,
    # AudioClassification) by using four different datasets. The two datasets
    # for regression are organized in csv format because that is the only
    # acceptable format for regression datasets. The two classification
    # datasets  are both organized in hierarchical format (even if also csv
    # format would be acceptable) to showcase the additional handling of
    # hierarchical format done by the classification classes.
    # Datasets will be loaded eagerly once for images and once for audios
    # to showcase that functionality, but the rest of the loading will be done
    # lazily because much more computationally manageable.

    # # # # # # # # #
    # IMAGE DATASET #
    # # # # # # # # #

    # IMAGE REGRESSION #
    img_regr_dataset = ImageRegressionDataset()
    img_regr_root = "yourpath/oop-final-project-group-7/image_regression_csv/images_poly"
    img_regr_labels_path = (
        "yourpath/oop-final-project-group-7/image_regression_csv/poly_targets_regression.csv"
    )

    # Eager loading without labels
    img_regr_dataset.load_data(
        root=img_regr_root, strategy="eager", format="csv"
    )
    image = img_regr_dataset[1]
    image = Image.fromarray(image)
    image.show()

    # Lazy loading with labels
    img_regr_dataset.load_data(
        root=img_regr_root,
        strategy="lazy",
        format="csv",
        labels_path=img_regr_labels_path,
    )
    image, target = img_regr_dataset[1]  # image is different than before
    # because the images are now ordered based on the csv file order
    image = Image.fromarray(image)
    image.show()
    print(f"Degrees of rotation of the polygon: {target}")

    # Splitting in train and test
    img_regr_train, img_regr_test = img_regr_dataset.train_test_split(
        train_size=0.6, shuffle=False  # shuffle is set based on user needs
    )
    print(f"Size of the image regression dataset: {len(img_regr_dataset)}")
    print(f"Size of the train dataset: {len(img_regr_train)}")
    print(f"Size of the test dataset: {len(img_regr_test)}")

    # IMAGE CLASSIFICATION #
    img_clas_dataset = ImageClassificationDataset()
    img_clas_root = "yourpath/oop-final-project-group-7/image_classification_hierarchy"

    # Lazy loading (the labels are automatically generated)
    img_clas_dataset.load_data(
        root=img_clas_root, strategy="lazy", format="hierarchical"
    )
    image, target = img_clas_dataset[1]
    image = Image.fromarray(image)
    image.show()
    print(f"The target is: {target}")

    # Splitting in train and test
    img_clas_train, img_clas_test = img_clas_dataset.train_test_split(
        train_size=0.6, shuffle=True
    )
    print(f"Size of the image classification dataset: {len(img_clas_dataset)}")
    print(f"Size of the train dataset: {len(img_clas_train)}")
    print(f"Size of the test dataset: {len(img_clas_test)}")

    # # # # # # # # #
    # AUDIO DATASET #
    # # # # # # # # #

    # AUDIO REGRESSION #

    aud_regr_dataset = AudioRegressionDataset()
    aud_regr_root = "yourpath/oop-final-project-group-7/audio_regression_csv/TRAIN"
    aud_regr_labels_path = "yourpath/oop-final-project-group-7/audio_regression_csv/TRAIN.csv"

    # Eeager loading without targets
    aud_regr_dataset.load_data(
        root=aud_regr_root,
        strategy="lazy",
        format="csv",
        labels_path=None,
    )

    # Show data point without target and play sound
    print(aud_regr_dataset[3])
    sample, sr = aud_regr_dataset[3]
    sd.play(sample, sr)
    sd.wait()

    # Reload lazily with targets
    aud_regr_dataset.load_data(
        root=aud_regr_root,
        strategy="lazy",
        format="csv",
        labels_path=aud_regr_labels_path,
    )

    # Show data point with target and play sound
    print(aud_regr_dataset[3])
    audio, target = aud_regr_dataset[3]
    sample, sr = audio
    sd.play(sample, sr)
    sd.wait()
    print(f"The target is:' {target}'")

    # Splitting in train and test
    aud_regr_train, aud_regr_test = aud_regr_dataset.train_test_split(
        train_size=0.6, shuffle=False
    )
    print(f"Size of the audio regression dataset: {len(aud_regr_dataset)}")
    print(f"Size of the train dataset: {len(aud_regr_train)}")
    print(f"Size of the test dataset: {len(aud_regr_test)}")

    # AUDIO CLASSIFICATION #
    aud_clas_dataset = AudioClassificationDataset()
    aud_clas_root = "yourpath/oop-final-project-group-7/audio_classification_hierarchy"

    # Lazy loading (targets are generated automatically)
    aud_clas_dataset.load_data(
        root=aud_clas_root, strategy="lazy", format="hierarchical"
    )

    # Show data point and play audio
    print(aud_clas_dataset[3])
    audio, target = aud_clas_dataset[3]
    sample, sr = audio
    sd.play(sample, sr)
    sd.wait()
    print(f"The target is:' {target}'")

    # Train and test split
    aud_clas_train, aud_clas_test = aud_clas_dataset.train_test_split(
        train_size=0.6, shuffle=True
    )
    print(f"Size of the audio classification dataset: {len(aud_clas_dataset)}")
    print(f"Size of the train dataset: {len(aud_clas_train)}")
    print(f"Size of the test dataset: {len(aud_clas_test)}")

    # # BATCHLOADING # #
    # In the next section we show the functionality of the BatchLoader class.
    # Since Batchloading works exactly the same, no matter how the data was
    # loaded, we show batchloader functionality with only one image dataset
    # and one audio dataset.
    # We showcase sequential batching with the image dataset and random
    # batching with the audio dataset.
    # We showcase the decision to not discard the last batch with the image
    # dataset, and the opposite decision with the audio dataset.
    # The dataset chosen to showcase this functionality train datasets
    # obtained by splitting the original bigger datasets because batchloading
    # is usually deployed during training of models, but it can work with any
    # of the original and any of the train and test datasets.

    # IMAGE BATCHLOADER IN SEQUENTIAL FASHION WITHOUT DISCARDING
    img_batcher = BatchLoader()
    img_batcher.create_batches(
        dataset=img_regr_train,
        batch_size=130,
        batch_style="sequential",
        discard_last_batch=False,
    )

    print(f"Number of batches of images created: {len(img_batcher)}")
    # 6000/130=46.2=47 becasue discard_last_batch=False

    img_batch = next(img_batcher)
    print(f"Length of the first batch is: {len(img_batch)}")
    # 130

    last_batch = None
    for img_batch in img_batcher:
        last_batch = img_batch
    print(f"Length of the last batch is: {len(last_batch)}")
    # 20 because 6000-(130*46)=20

    # AUDIO BATCHLOADER IN RANDOM FASHION WITH DISCARDING
    aud_batcher = BatchLoader()
    aud_batcher.create_batches(
        dataset=aud_clas_train,
        batch_size=100,
        batch_style="random",
        discard_last_batch=True,
    )

    print(f"Number of batches created: {len(aud_batcher)}")
    # 630/100=6.3=6 because discard_last_batch=True

    aud_batch = next(aud_batcher)
    print(f"Length of the first batch is: {len(aud_batch)}")
    # 100

    last_batch = None
    for aud_batch in aud_batcher:
        last_batch = aud_batch
    print(f"Length of the last batch is: {len(last_batch)}")
    # 100 because after duscarding, the last batch has the same length of
    # every other batch

    # # PIPELINES # #
    # In the next section we show the functionality of the preprocessing
    # classes in combination with the preprocessiing pipeline class. Again,
    # since preprocessing works the same independently of how the data was
    # originally loaded and of whether there are targets or not, we show one
    # preprocessing pipeline for images and one preprocessing pipeline for
    # audios

    # IMAGE PREPROCESSING PIPELINE
    image, target = img_regr_dataset[5]
    image_show = Image.fromarray(image)
    image_show.show()  # show original image

    # define pipeline
    crop = ImageCenterCrop(70, 100)
    patch = ImagePatching("red", 30, 30)
    img_pipeline = PreprocessingPipeline(crop, patch)

    # apply pipeline
    preprocessed_image = img_pipeline(image)
    preprocessed_image_show = Image.fromarray(preprocessed_image)
    preprocessed_image_show.show()  # show preprocessed image

    # AUDIO PREPROCESSING PIPELINE
    audio, target = aud_clas_dataset[10]

    # define pipeline
    pitch_shift = PitchShifting(100)
    resample = Resampling(6000)
    aud_pipeline = PreprocessingPipeline(pitch_shift, resample)

    # apply pipeline
    processed_audio_samples, processed_audio_s_rate = aud_pipeline(audio)

    # play audio after pitch shifting and resampling
    print("Audio signal after preprocessing steps were applied")
    sd.play(processed_audio_samples, processed_audio_s_rate)
    sd.wait()


if __name__ == "__main__":
    main()
