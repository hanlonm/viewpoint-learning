import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
        padding = np.full((pad_length, item.shape[1]), 0)

        
        padded_item = np.vstack((item, padding))
        padded_batch.append(padded_item)
        
        mask = len(item) * [False] + pad_length * [True]
        masks.append(mask)
        labels.append(label)
        
    return torch.Tensor(np.array(padded_batch)), torch.Tensor(np.array(masks)), torch.Tensor(np.array(labels))


class TestCallback(pl.Callback):
    def __init__(self, test_loader):
        #self.best_val_loss = float('inf')
        print()
        self.test_loader = test_loader

    def on_epoch_end(self, trainer:pl.Trainer, pl_module):
        # Run evaluation on test set
        trainer.test(model=pl_module, dataloaders=self.test_loader)
        #print(f"Test loss: {test_loss:.4f}")

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, collate_fn=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def prepare_data(self):
        # Any data download or preparation can be done here
        # This method is called only once on a single GPU
        pass

    def setup(self, stage=None):
        # Split your datasets or perform any setup steps here
        # This method is called on every GPU in distributed training
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
