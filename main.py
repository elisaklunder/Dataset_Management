import os
import sys

sys.path.append(os.getcwd() + "/src/")
from src.image_class import ImageDataset


def main():
    poly_dataset = ImageDataset()
    root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7"
    labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\poly_targets_classification.csv"
    poly_dataset.load_data(root_path, "eager", "csv", labels_path)
    print(len(poly_dataset))

if __name__ == "__main__":
    main()
