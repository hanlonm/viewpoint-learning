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
    

class TransformerDataset(Dataset):
    def __init__(self, viewpoint_sequences, labels):
        super(TransformerDataset, self).__init__()
        self.viewpoint_sequences = viewpoint_sequences
        self.labels = labels

    def __getitem__(self, idx):
        viewpoint_sequence = self.viewpoint_sequences[idx]
        label = self.labels[idx]

        return viewpoint_sequence, label

    def __len__(self):
        return self.labels.shape[0]
    
def transformer_collate(batch):
    max_seq_len = max(len(item) for item, _ in batch)
    padded_batch = []
    masks = []
    labels = []
    
    for item, label in batch:
        pad_length = max_seq_len - len(item)
        padding = np.full((pad_length, item.shape[1]), -1)

        
        padded_item = np.vstack((item, padding))
        padded_batch.append(padded_item)
        
        mask = len(item) * [False] + pad_length * [True]
        masks.append(mask)
        labels.append(label)
        
    return torch.Tensor(np.array(padded_batch)), torch.Tensor(np.array(masks)), torch.Tensor(np.array(labels))