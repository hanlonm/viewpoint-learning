import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.r2 import R2Score
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch import optim, nn, utils, Tensor
from typing import Any



class NaivePCTCls(nn.Module):
    def __init__(self, num_categories=2):
        super().__init__()

        self.encoder = NaivePCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(73+0-64, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        # self.sa3 = SA(128)
        # self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Define the layers for the 2D pathway
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        # self.linear2d = nn.Linear(in_features=512, out_features=64)
    
    def forward(self, x: torch.Tensor):
        B, N, T = x.shape
        # heatmaps = x[:,:, -64:]
        # heatmaps = heatmaps.view(B*N, 1, 8, 8)
        # heatmaps = F.relu(self.conv2d(heatmaps))
        # heatmaps = F.relu(self.linear2d(heatmaps))
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        # x3 = self.sa3(x2)
        # x4 = self.sa4(x3)
        x = torch.cat([x1, x2], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean
    

class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(128, 128, bias=False)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_categories)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        self.dp1 = nn.Dropout(p=0.4)
        self.dp2 = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    



class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = x.permute(0, 2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    

class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        has_nan = torch.isnan(energy).any().item()
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x + x_s

        return x
    



class PCTViewpointTransformer(pl.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = NaivePCTCls()
        # self.example_input_array = next(iter(train_loader))[0]

        #self.acc_metric = MeanAbsoluteError()
        self.variances = torch.tensor(
            [0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 5, 5] + 64 * [2] + 3 * [0.1]) / 10
        self.variances = torch.sqrt(self.variances).cuda()

        self.normalizer = torch.tensor([5,5,5]+ 4*[6.3] + [1280, 720]).cuda()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x / self.normalizer
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        # optimizer = optim.Adam(self.parameters())
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[500],
                                                      gamma=10)
        # lr_scheduler = Scheduler(optimizer, 256,4000)
        return [optimizer], [lr_scheduler]

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.variances

        return x + noise

    def _calculate_loss(self, batch, mode="train", prog_bar=False):
        tokens, labels = batch
        # if mode == "train":
        #     tokens = self.add_noise(tokens)
        tokens = tokens / self.normalizer
        has_nan = torch.isnan(tokens).any().item()

        logits = self.model(tokens)
        labels = labels.squeeze(dim=1).long()
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, prog_bar=prog_bar)
        # self.log("%s_r2" % mode, r2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self._calculate_loss(batch, mode="val", prog_bar=True)
        else:
            self._calculate_loss(batch, mode="val", prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
