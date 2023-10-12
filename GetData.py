from torch.utils.data.dataset import Dataset
import numpy as np

class GetData(Dataset):
    def __init__(self, X_train, y_train, Transform):
        self.X = X_train
        self.y = y_train
        self.transform = Transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        image = self.transform(self.X[index]).permute((1, 2, 0)).contiguous()
        label = self.y[index]
        return image, label