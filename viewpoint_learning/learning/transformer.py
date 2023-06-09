import os
from typing import Any
import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.r2 import R2Score
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer




class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    


class ViewpointTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()


        # Layers/Networks
        self.input_layer = nn.Linear(73, embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x:torch.Tensor):
        # Preprocess input
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    
    
class ViewpointTransformerReg(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()


        # Layers/Networks
        self.input_layer = nn.Linear(73, embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x:torch.Tensor):
        # Preprocess input
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    

class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViewpointTransformer(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]

        #self.acc_metric = MeanAbsoluteError()
        self.r2_score = R2Score()

        self.loss = nn.BCELoss()



    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        tokens, labels = batch
        preds = self.model(tokens)
        # print(preds)
        # print(labels)
        loss = self.loss(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        acc = accuracy(preds, labels,task='binary')
        # acc = self.acc_metric(preds, labels)
        # r2 = self.r2_score(preds, labels)

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, prog_bar=True)
        # self.log("%s_r2" % mode, r2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class ViTReg(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViewpointTransformerReg(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]

        self.acc_metric = MeanAbsoluteError()
        self.r2_score = R2Score()

        self.loss = nn.SmoothL1Loss(reduction="mean")



    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        tokens, labels = batch
        preds = self.model(tokens)
        # print(preds)
        # print(labels)
        loss = self.loss(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        acc = self.acc_metric(preds, labels)
        # acc = self.acc_metric(preds, labels)
        r2 = self.r2_score(preds, labels)

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, prog_bar=True)
        self.log("%s_r2" % mode, r2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")




class TransformerModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        encoding_dim = 128
        # self.embedding = nn.Embedding(73, 512)
        self.embedding = nn.Linear(73, encoding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoding_dim))

        # self.transformer = nn.Transformer(
        #     d_model=128,
        #     nhead=2,
        #     num_encoder_layers=4,
        #     num_decoder_layers=4
        # )
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_dim, nhead=2, dim_feedforward=64, norm_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_dim, nhead=8, dim_feedforward=2048, norm_first=True, dropout=0.1)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(encoding_dim, 2)

    def forward(self, x, mask):
        n_non_padded = torch.sum(mask)
        embedded = self.embedding(x)
        # idx = torch.randperm(embedded.shape[1])
        # embedded = embedded[:, idx]
        # mask = mask[:, idx]
        # Add CLS token
        B, T, _ = embedded.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        embedded = torch.cat([cls_token, embedded], dim=1)
        cls_mask = torch.zeros(B,1).to(embedded.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        embedded = embedded.permute(1, 0, 2)  # Transpose to (seq_len, batch_size, hidden_dim)
        hidden = self.encoder(embedded, src_key_padding_mask=mask)
        hidden = hidden.permute(1, 0, 2)  # Transpose back to (batch_size, seq_len, hidden_dim)
        logits = self.fc(hidden[:, 0, :])  # Use the last hidden state for classification
        return logits

class ViewpointTransformer(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = TransformerModel()
        # self.example_input_array = next(iter(train_loader))[0]

        #self.acc_metric = MeanAbsoluteError()
        self.variances = torch.tensor([0.1,0.1,0.1,0.09,0.09,0.09,0.09,5,5] + 64 * [2])
        self.variances = torch.sqrt(self.variances).cuda()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-6)
        # optimizer = optim.Adam(self.parameters())
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=10)
        # lr_scheduler = Scheduler(optimizer, 256,4000)
        return [optimizer], [lr_scheduler]
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.variances

        return x + noise
    
    def _calculate_loss(self, batch, mode="train", prog_bar=False):
        tokens, masks,labels = batch
        if mode == "train":
            tokens = self.add_noise(tokens)
        logits = self.model(tokens, masks)
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
            self._calculate_loss(batch, mode="val",prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
        

    # def train_dataloader(self):
    #     return self.data_module.train_dataloader()

    # def val_dataloader(self):
    #     return self.data_module.val_dataloader()

    # def test_dataloader(self):
    #     return self.data_module.test_dataloader()

class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))