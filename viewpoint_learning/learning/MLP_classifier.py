import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
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
                                   nn.Linear(8, 1),
                                   nn.Sigmoid())
        self.trainable_parameters = self.model.parameters()

        self.loss = nn.BCELoss()

    def forward(self, x):
        out = self.model(x)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        self.log("val_loss", test_loss)
        acc = accuracy(y_hat, y,task='binary')
        self.log("acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainable_parameters, lr=1e-3)
        return optimizer