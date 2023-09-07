import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.functional import accuracy



# define the LightningModule
class ViewpointClassifier(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
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
            nn.Linear(8, 2)
        )
        self.trainable_parameters = self.model.parameters()

        self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        B, D = x.shape
        x: torch.Tensor = x.unsqueeze(1)
        x = self.bn(self.conv1d_1(x))
        x = self.bn(self.conv1d_2(x))
        x = x.reshape(B, -1)
        out = self.model(x)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x + (0.0**0.5)*torch.randn_like(x)
        B, D = x.shape
        x: torch.Tensor = x.unsqueeze(1)
        x = self.bn(self.conv1d_1(x))
        x = self.bn(self.conv1d_2(x))
        x = x.reshape(B, -1)
        y = y.squeeze(dim=1).long()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        B, D = x.shape
        x: torch.Tensor = x.unsqueeze(1)
        x = self.bn(self.conv1d_1(x))
        x = self.bn(self.conv1d_2(x))
        x = x.reshape(B, -1)
        logits = self.model(x)

        y = y.squeeze(dim=1).long()
        loss = self.loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        B, D = x.shape
        x: torch.Tensor = x.unsqueeze(1)
        x = self.bn(self.conv1d_1(x))
        x = self.bn(self.conv1d_2(x))
        x = x.reshape(B, -1)
        y = y.squeeze(dim=1).long()
        logits = self.model(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainable_parameters, lr=1e-5)
        

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
        
        # Specify the monitor metric to track for reducing the learning rate
        monitor_metric = 'val_loss/dataloader_idx_1'  # Replace with your chosen metric
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': monitor_metric
        }
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
            