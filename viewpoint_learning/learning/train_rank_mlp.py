import os
from dataloader import RankDataset
from torch.utils.data import DataLoader, random_split
from rank_mlp import RankMLP
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py
import matplotlib.pyplot as plt
from utils import normalize, standardize, create_dataset, create_rank_dataset, create_rank_trf_dataset

home_dir = os.environ.get("BASE_DIR", "/local/home/hanlonm")

hf = h5py.File(str(home_dir)+"/active-viewpoint-selection/data/training_data/opt_occ_100_50_230724.h5", "r")

print(hf.keys())
train_environments = ["00067_opt", "00596_opt", "00638_opt", "00700_opt", "00269_opt"]
test_environments = ["00195_opt", "00654_opt", "00111_opt", "00403_opt"]

batch_size = 16
lr = 1e-4
name = "test_100_opt_bal_weight"
input_config = f"{name}_bs{batch_size}_lr{lr}"


max_error = 5.0

train_data = create_rank_dataset(hf, train_environments, max_error, samples_per_point=100)
test_data = create_rank_dataset(hf, train_environments, max_error, samples_per_point=50)


train_dataset = RankDataset(train_data)
test_dataset = RankDataset(test_data)

# use 20% of training data for validation
train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size])

train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, 8, shuffle=False, num_workers=0)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss/dataloader_idx_0", mode="min",save_weights_only=True)
checkpoint_test_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss/dataloader_idx_1", mode="min",save_weights_only=True,filename='best_test')

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"MLP_rank/{input_config}")

model = RankMLP(lr=lr)

trainer = pl.Trainer(max_epochs=400, logger=tb_logger, callbacks=[checkpoint_callback,checkpoint_test_callback])
trainer.fit(model,train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader])
trainer.test(dataloaders=test_loader, ckpt_path=f"best{input_config}")

