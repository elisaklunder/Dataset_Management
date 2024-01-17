import os
import sys

sys.path.append(os.getcwd() + "/src/")
from src.image_classification_dataset import ImageClassificationDataset


def main():
    poly_dataset = ImageClassificationDataset()
    
    # JULIA
    #root_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_hierarchy"
    
    # ELI 
    #root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\images_poly"
    root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_hierarchy"
    #labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\poly_targets_classification.csv"
    poly_dataset.load_data(root_path, "lazy", "hierarchical")
    train, test = poly_dataset.train_test_split(
        train_size=0.6, test_size=0.4, shuffle=True
    )
    print(poly_dataset[4])


if __name__ == "__main__":
    main()
