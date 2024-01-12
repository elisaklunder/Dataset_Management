Image(root adjfnadkbjfa fafb)


class BaseDataset:
    def __init__(self, root: str, labels: bool = False, format: str = "csv", strategy: str):
        self._root = root #is this private? cause the user shouldn't access it after the root was set
        self.labels_bool = labels
        self.format = format
        self._data = [] #getters and setters
        self._labels = []


     #####THIS COULD BE A CLASS THAT IS CALLED IN HERE######
        #with setters we could set the self._data  and the self_labels if needed
    def _lazy_load_data(self):
        #batches + let user chose about the last batch
        
        if not self.labels:
            self.data = #read data in root folder
        else:
            if format == "csv":
                self.data = #read data in root folder
                self.labels = #read labels from csv outside of folder
            if format == "hierarchical":
                #smth fucking weird


    def _eager_load_data(self): 
        #load entire data set at once
        pass

    def load_data(self, strategy, format):
        if strategy == "eager":
            _eager_load_data(self)
        else:
            _lazy_load_data(self)
             


    def __getitem__(self, index:int):
        if bool(self._data):
            if bool(self._labels):
                return (self._labels[index], self._data[index])
            else:
                return self._data[index]
    
    def __len__(self):
        return len(self._data)
    
    def train_test_split(self, train_size, test_size ):
        pass
        