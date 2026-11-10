import torch
from torch.utils.data import Dataset, DataLoader

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

class Custom_Dataset(Dataset):
    """
    Class used to define a custom dataset

    Attributes
        X:    Matrix of features
        y:    Vector of labels
    """
    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y
       
    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index]
   
    def __len__(self):
        'Returns the total number of samples'
        return self.X.shape[0]

class Load_Dataset:

    def __init__(self, scaled_train, scaled_test, scaled_val, scaled_ytrain, scaled_ytest, scaled_yval):
        'Initialization'
        self.scaled_train  = scaled_train
        self.scaled_test   = scaled_test
        self.scaled_val    = scaled_val
        self.scaled_ytrain = scaled_ytrain
        self.scaled_ytest  = scaled_ytest
        self.scaled_yval   = scaled_yval
        
    def load_data(self):        
        # Convert the NumPy arrays to PyTorch tensors
        X1 = torch.from_numpy(self.scaled_train).to(dtype=torch.float32).to(device)
        y1 = torch.from_numpy(self.scaled_ytrain).to(dtype=torch.float32).to(device)
        X2 = torch.from_numpy(self.scaled_test).to(dtype=torch.float32).to(device)
        y2 = torch.from_numpy(self.scaled_ytest).to(dtype=torch.float32).to(device)
        X3 = torch.from_numpy(self.scaled_val).to(dtype=torch.float32).to(device)
        y3 = torch.from_numpy(self.scaled_yval).to(dtype=torch.float32).to(device)

        train_dataset = Custom_Dataset(X1, y1)
        test_dataset = Custom_Dataset(X2, y2)
        val_dataset = Custom_Dataset(X3, y3)

        return train_dataset, test_dataset, val_dataset