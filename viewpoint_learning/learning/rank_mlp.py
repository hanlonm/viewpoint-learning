import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.functional import accuracy


class RankMLP(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.bn = nn.MaxPool1d(kernel_size=2, stride=2)

        self.model = nn.Sequential(
            #nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Linear(32*36, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.trainable_parameters = self.model.parameters()

        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()


    def forward(self, x):
        B, D = x.shape
        x: torch.Tensor = x.unsqueeze(1)
        x = self.bn(self.conv1d_1(x))
        x = self.bn(self.conv1d_2(x))
        x = x.reshape(B, -1)
        out = self.model(x)

        return out
    
    def _calculate_loss(self, batch, mode="train", prog_bar=False):
        x1, x2, y, weights = batch
        B, D = x1.shape
        x1: torch.Tensor = x1.unsqueeze(1)
        x1 = self.bn(self.conv1d_1(x1))
        x1 = self.bn(self.conv1d_2(x1))
        x1 = x1.reshape(B, -1)
        out_1 = self.model(x1)

        B, D = x2.shape
        x2: torch.Tensor = x2.unsqueeze(1)
        x2 = self.bn(self.conv1d_1(x2))
        x2 = self.bn(self.conv1d_2(x2))
        x2 = x2.reshape(B, -1)
        out_2 = self.model(x2)

        y_hat = out_1 - out_2
        loss = nn.BCEWithLogitsLoss(weight=weights)(y_hat, y)
        # Logging to TensorBoard by default
        self.log(f"{mode}_loss", loss, prog_bar=prog_bar)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainable_parameters, lr=self.lr)
        
        # optimizer = optim.Adam(self.parameters())
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[1000],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]
    
    def on_train_epoch_end(self):

        #  the function is called after every epoch is completed
        if(self.current_epoch==0):
            sampleImg=torch.rand((1,146)).cuda()
            self.logger.experiment.add_graph(self,sampleImg)
        self.custom_histogram_adder()

    def custom_histogram_adder(self):
        
        # iterating through all parameters
        for name,params in self.named_parameters():
           
            self.logger.experiment.add_histogram(name,params,self.current_epoch)