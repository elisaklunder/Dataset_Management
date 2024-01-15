import os
import sys

sys.path.append(os.getcwd() + "/src/")
from src.image_dataset import ImageDataset
from src.img_class import ImageClassification


def main():
    poly_dataset = ImageClassification()
    #root_path_hierarchy = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_hierarchy"
    root_path_csv = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\images_poly"
    # root_path_hierarchy = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_hierarchy"
    labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\poly_targets_classification.csv"
    poly_dataset.load_data(root_path_csv, "lazy", "csv", labels_path)
    train, test = poly_dataset.train_test_split(
        train_size=0.6, test_size=0.4, shuffle=True
    )
    print(type(train), type(test))


if __name__ == "__main__":
    main()
