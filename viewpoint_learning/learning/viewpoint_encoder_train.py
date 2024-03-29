from dataloader import ViewpointDataset
from torch.utils.data import DataLoader, random_split
from viewpoint_encoder import ViewpointAutoEncoder
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py


hf = h5py.File("/local/home/hanlonm/active-viewpoint-selection/data/00195_HL_SPA_NN/test-2000.h5", "r")
print(hf.keys())
histogram_data: np.ndarray = hf["histogram_data"][:]
num_points = hf.attrs["num_points"]
num_angles = hf.attrs["num_angles"]
labels = hf["errors"][:]
histogram_data = histogram_data.reshape(
    (num_points * num_angles, histogram_data.shape[2]))
labels: np.ndarray = labels.reshape((num_points * num_angles, labels.shape[2]))
e_trans = np.array([np.linalg.norm(labels[:,:3], axis=1)]).T
e_rot = np.array([labels[:,3]]).T

labels = np.hstack((e_trans, e_rot))

dataset = ViewpointDataset(histogram_data, labels)

# use 10% of training data for validation
train_set_size = int(len(dataset) * 0.9)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, 8, shuffle=True, num_workers=8)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=8)

checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="val_loss", save_weights_only=True)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="AutoEncoder/")
model = ViewpointAutoEncoder(histogram_data.shape[1], 64)
trainer = pl.Trainer(max_epochs=100, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)