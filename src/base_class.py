class BaseDataset:
    def __init__(self, root: str, labels: bool = False, format: str = "csv", strategy: str):
        self.root = root
        self.labels = labels
        self.format = format
        self.data = []
        self.labels = []
        
        
    def _load_data(self):
        
        if not self.labels:
            self.data = #read data in root folder
        else:
            if format == "csv":
                self.data = #read data in root folder
                self.labels = #read labels from csv outside of folder
            if format == "hierarchical":
                #smth fucking weird     

    def __getitem__(self, index:int):
        if self.labels:
            return
    
    def __len__(self):
        