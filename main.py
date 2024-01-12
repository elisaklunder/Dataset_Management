import os
import sys

sys.path.append(os.getcwd() + "/src/")
# from example import ExampleClass


def main():
    poly_dataset = BaseDataset()
    root = "C:\Users\elikl\Documents\Università\yr2\2 - OOP\targets.csv"
    poly_data = poly_dataset.load_data(root)

if __name__ == "__main__":
    main()
