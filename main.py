import os
import sys

sys.path.append(os.getcwd() + "/src/")
from src.image_classification_dataset import ImageClassificationDataset
from src.batchloader import BatchLoader


def main():
    poly_dataset = ImageClassificationDataset()
    
    # JULIA
    #root_path = "/Users/juliabelloni/Desktop/oop/assignments/oop-final-project-group-7/classification_hierarchy"
    
    # ELI 
    root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\images_poly"
    #root_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_hierarchy"
    labels_path = r"C:\Users\elikl\Documents\Università\yr2\2 - OOP\oop-final-project-group-7\classification_csv\poly_targets_classification.csv"
    poly_dataset.load_data(root_path, "lazy", "csv", labels_path)
    train, test = poly_dataset.train_test_split(
        train_size=0.6, test_size=0.4, shuffle=False
    )
    print(train[4])
    print(len(train))
    
    batcher = BatchLoader()
    
    batcher.create_batches(train, 51, "shuffle", False)
    print(len(batcher))


if __name__ == "__main__":
    main()
