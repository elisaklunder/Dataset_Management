import random
from typing import Iterator, List

from base_dataset import BaseDataset
from errors import Errors


class BatchLoader:
    def __init__(self) -> None:
        """
        Constructor of the class.
        """
        self._batch_size = None
        self._dataset = None
        self._batches = []
        self._index = 0
        self._errors = Errors()

    def __iter__(self) -> Iterator:
        """
        Magic method implementing __iter__ for the object

        Returns:
            Iterator: An iterator object for iterating over batches.

        """
        self._index = 0
        return self

    def __next__(self) -> List:
        """
        Magic method implementing the retrieval of the next batch of items
        from the dataset.

        Raises:
            StopIteration: If attempting to access a batch beyond the
            dataset's length.

        Returns:
            List: list containing the indexes of the items to be loaded.
        """
        if self._index >= len(self):
            raise StopIteration
        batch = []
        for element in self._batches[self._index]:
            batch.append(self._dataset[element])
        self._index += 1
        return batch

    def create_batches(
        self,
        dataset: BaseDataset,
        batch_size: int,
        batch_style: str,
        discard_last_batch: bool,
    ) -> None:
        """
        Method that creates batches given a specific batch size.

        Args:
            dataset (BaseDataset): The dataset on which batch creation is
            to be carried out.

            batch_size (int): integer number indicating the number of indexes
            in each batch

            batch_style (str): the batch creation can either be 'sequential' or
            'random'. In the latter case the batches of data are created by
            randomly shuffling the order of the data points

            discard_last_batch (bool): bool indicating whether to discard the
            last batch if it is smaller than the others.

        Raises:
            TypeError: if the dataset argument is not an instance of the class
            BaseDataset (or a subclass of this class).
            TypeError: if the batch_size argument is not an int.
            TypeError: if the batch_style argument is not a string.
            TypeError: if the discard_last_batch argument is no a boolean
            value.
            ValueError: if the batch_size argument is negative.
            ValueError: if the batch_style argument is not one of the expected
            strings.

        """
        self._errors.type_check("dataset", dataset, BaseDataset)
        self._errors.type_check("batch_size", batch_size, int)
        self._errors.ispositive("batch_size", batch_size)
        self._errors.type_check("batch_style", batch_style, str)
        self._errors.value_check(
            "batch_style", batch_style, "random", "sequential"
        )
        self._errors.type_check("discard_last_batch", discard_last_batch, bool)

        self._batch_size = batch_size
        self._dataset = dataset

        indexes = list(range(len(dataset)))
        if batch_style == "random":
            indexes = random.sample(indexes, len(indexes))

        for i in range(0, len(dataset), batch_size):
            self._batches.append(indexes[i : i + batch_size])

        if len(dataset) % batch_size != 0 and discard_last_batch:
            self._batches.pop()

    def __len__(self) -> int:
        """
        Magic method implementing len() function for the object.

        Returns:
            int: integer number indicating the number the number of batches
            that can be created from the dataset with the specific batch size.
        """
        return len(self._batches)
