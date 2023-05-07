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

        return viewpoint.flatten(), error

    def __len__(self):
        return self.errors.shape[0]