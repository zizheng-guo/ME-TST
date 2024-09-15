
from torch.utils.data import Dataset
import random

# Convert dataset into tensor for training and testing
class OFFSTRDataset(Dataset):
    def __init__(self, tensors, transform, train):
        self.tensors = tensors
        self.train = train
        self.transform = transform
        
    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        if self.train:
            val = random.randint(90, 110) / 100
            x = self.tensors[0][index] * val
        else:
            x = self.tensors[0][index]
        y  = self.tensors[1][index]
        y1 = self.tensors[2][index]
        return x, y, y1
    