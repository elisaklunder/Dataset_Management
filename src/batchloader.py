

class BatchLoader():
    def __init__(self):
        self._batch_size = None
        self._fashion = None
        self._data = None
        # train data is a dataset considering that strategy has a getter

    def _single_batch_load(self):
        for i in range(self._batch_size):
                data = self._data[i:i + self._batch_size]
            pass
    
    def load_batches(self, train_data, batch_size, fashion):
        self._batch_size = batch_size
        self._fashion = fashion
        self._data = train_data.data
        if len(train_data)%batch_size == 0:
            self._single_batch_load()
        else:
            #ask user if they want to use last batch of the train
            print("the last batch is smaller, type yes if you still want to use the last batch, type no otherwise?")
            user = input()
            if user == "yes":
                for i in range(self._batch_size -1):
                    #case for the last batch
                    if i == self._batch_size - 1:
                        data = self._data[i:]
                    self._single_batch_load
            elif user == "no":
                for i in range(self._batch_size -1):
                    data = self._data[i:i + self._batch_size]
            pass

    def __len__(self):
        print(f"number of batches that can be created using {self._batch_size} as batch size")
        return self._data//self._batch_size
                 