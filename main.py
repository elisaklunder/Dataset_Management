import os
import sys

sys.path.append(os.getcwd() + "/src/")
from src.image_dataset import ImageDataset


def main():
    poly_dataset = ImageDataset()
    root_path_csv = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\images_poly"
    root_path_hierarchy = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_hierarchy"
    labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\poly_targets_classification.csv"
    poly_dataset.load_data(root_path_csv, "lazy", "csv")
    train, test = poly_dataset.train_test_split(
        train_size=0.6, test_size=0.4, shuffle=True
    )
    print(len(train))
    print(len(test))
    print(poly_dataset[4])


if __name__ == "__main__":
    main()
