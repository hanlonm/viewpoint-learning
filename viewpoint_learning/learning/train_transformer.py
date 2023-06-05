from dataloader import ClassifierDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from transformer import ViT
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py
import matplotlib.pyplot as plt


from utils import normalize, standardize, pre_process, remove_nan_rows, create_transformer_dataset

hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/token_test_1.h5", "r+")
#hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/230522_100.h5", "r+")
print(hf.keys())
num_points = hf.attrs["num_points"]
num_angles = hf.attrs["num_angles"]

input_config = "token_1024"

train_environments = ["00067", "00596", "00638", "00700"]
test_environments = ["00195", "00654"]

max_error = 5.0
train_tokens, train_trans_errors, train_rot_errors = create_transformer_dataset(hf, train_environments, max_error)
test_tokens, test_trans_errors, test_rot_errors = create_transformer_dataset(hf, test_environments, max_error)

train_errors = np.hstack((train_trans_errors, train_rot_errors))
train_labels = np.logical_and((train_errors[:, 0]*max_error) < 0.05, train_errors[:, 1] < 0.5)
train_labels = train_labels.astype(int)
train_labels = np.array([train_labels]).T

test_errors = np.hstack((test_trans_errors, test_rot_errors))
test_labels = np.logical_and((test_errors[:, 0]*max_error) < 0.05, test_errors[:, 1] < 0.5)
test_labels = test_labels.astype(int)
test_labels = np.array([test_labels]).T

pos = np.argwhere(train_labels==1)[:,0]
neg = np.argwhere(train_labels==0)[:,0]
pos = np.random.choice(pos, 400)
neg = np.random.choice(neg, 400)

pos_toks = train_tokens[pos]
neg_toks = train_tokens[neg]
pos_labels = train_labels[pos]
neg_labels = train_labels[neg]

input_data = np.vstack((pos_toks, neg_toks))
train_labels = np.vstack((pos_labels, neg_labels))

tot = np.sum(train_labels)
print(np.sum(test_labels)/len(test_labels))

train_dataset = ClassifierDataset(input_data, train_labels)
test_dataset = ClassifierDataset(test_tokens, test_labels)

# use 10% of training data for validation
train_set_size = int(len(train_dataset) * 0.9)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, 8, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, 8, shuffle=False, num_workers=0)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max",save_weights_only=True)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"Transformer/{input_config}")
model = ViT(model_kwargs={
        "embed_dim": 80,
        "hidden_dim": 160,
        "num_heads": 1,
        "num_layers": 1,
        "num_classes": 2,
        "dropout": 0.2,
    },
    lr=3e-4)
trainer = pl.Trainer(max_epochs=200, logger=tb_logger, callbacks=[checkpoint_callback, LearningRateMonitor("epoch")])
trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
trainer.fit(model, train_loader, val_loader)
trainer.test(dataloaders=test_loader, ckpt_path="best")
