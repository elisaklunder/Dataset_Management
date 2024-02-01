import random
from typing import Iterator, List

from base_dataset import BaseDataset
from errors import Errors


class BatchLoader:
    def __init__(self) -> None:
        self._batch_size = None
        self._dataset = None
        self._batches = []
        self._index = 0
        self._errors = Errors()

    def __iter__(self) -> Iterator:
        self._index = 0
        return self

    def __next__(self) -> List:
        if self._index >= len(self):
            raise StopIteration
        batch = []
        for element in self._batches[self._index]:
            batch.append(self._dataset[element])
        self._index += 1
        return batch

    def create_batches(
        self,
        train_dataset: BaseDataset,
        batch_size: int,
        batch_style: str,
        discard_last_batch: bool,
    ) -> None:
        self.errors.type_check("train_dataset", train_dataset, BaseDataset)
        self.errors.type_check("batch_size", batch_size, int)
        self.errors.ispositive("batch_size", batch_size)
        self.errors.type_check("batch_style", batch_style, str)
        self.errors.value_check(
            "batch_style", batch_style, "random", "sequential"
        )
        self.errors.type_check("discard_last_batch", discard_last_batch, bool)

        self._batch_size = batch_size
        self._dataset = train_dataset

        indexes = list(range(len(train_dataset)))
        if batch_style == "random":
            indexes = random.sample(indexes, len(indexes))

        for i in range(0, len(train_dataset), batch_size):
            self._batches.append(indexes[i: i + batch_size])

        if len(train_dataset) % batch_size != 0 and discard_last_batch:
            self._batches.pop()

    def __len__(self) -> int:
        return len(self._batches)
