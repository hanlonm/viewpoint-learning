import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
import torch
from torchmetrics.functional import accuracy



# define the LightningModule
class ViewpointClassifier(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 96),
                                   nn.ReLU(),
                                   nn.Linear(96,64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 2))
        self.trainable_parameters = self.model.parameters()

        self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        out = self.model(x)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x + (0.1**0.5)*torch.randn_like(x)
        y = y.squeeze(dim=1).long()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y = y.squeeze(dim=1).long()
        logits = self.model(x)
        loss = self.loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y = y.squeeze(dim=1).long()
        logits = self.model(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainable_parameters, lr=1e-3)
        return optimizer