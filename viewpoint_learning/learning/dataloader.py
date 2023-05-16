import numpy as np
import torch
from torch.utils.data import Dataset

class ViewpointDataset(Dataset):
    def __init__(self, viewpoints, errors):
        super(ViewpointDataset, self).__init__()
        self.viewpoints = viewpoints
        self.errors = errors

    def __getitem__(self, idx):
        viewpoint = torch.Tensor(self.viewpoints[idx])
        error = torch.Tensor(self.errors[idx])

        return viewpoint, error

    def __len__(self):
        return self.errors.shape[0]
    

class ClassifierDataset(Dataset):
    def __init__(self, viewpoints, labels):
        super(ClassifierDataset, self).__init__()
        self.viewpoints = viewpoints
        self.labels = labels

    def __getitem__(self, idx):
        viewpoint = torch.Tensor(self.viewpoints[idx])
        label = torch.Tensor(self.labels[idx])

        return viewpoint, label

    def __len__(self):
        return self.labels.shape[0]
    
class RegressionDataset(Dataset):
    def __init__(self, viewpoints, values):
        super(RegressionDataset, self).__init__()
        self.viewpoints = viewpoints
        self.values = values

    def __getitem__(self, idx):
        viewpoint = torch.Tensor(self.viewpoints[idx])
        value = torch.Tensor(self.values[idx])

        return viewpoint, value

    def __len__(self):
        return self.values.shape[0]