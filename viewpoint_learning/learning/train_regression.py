from dataloader import RegressionDataset
from torch.utils.data import DataLoader, random_split
from regression import ViewpointRegressor
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py

from utils import normalize, standardize


hf = h5py.File("/local/home/hanlonm/mt-matthew/data/00195_HL_SPA_NN/test-2000.h5", "r+")
print(hf.keys())
histogram_data: np.ndarray = hf["histogram_data"][:]
num_points = hf.attrs["num_points"]
num_angles = hf.attrs["num_angles"]
errors = hf["errors"][:]
histogram_data = histogram_data.reshape(
    (num_points * num_angles, histogram_data.shape[2]))


# Ablation
ranges = histogram_data[:,:2]
range_1 = standardize(histogram_data[:, 0], 2000)
range_2 = standardize(histogram_data[:, 1], 2000)
min_dist_hist = histogram_data[:,2:12]
max_dist_hist = histogram_data[:,12:22]
min_ang_hist =histogram_data[:,22:32]
max_ang_hist = histogram_data[:,32:42]
min_ang_diff_hist =histogram_data[:,42:52]
max_ang_diff_hist =histogram_data[:,52:62]
heatmaps = histogram_data[:,62:126]
heatmaps = standardize(histogram_data[:,62:126], 1000)
px_u_hist=histogram_data[:,126:136]
px_v_hist=histogram_data[:,136:146]


histogram_data = np.hstack((np.array([range_1]).T, np.array([range_2]).T, min_dist_hist, max_dist_hist, min_ang_hist, max_ang_hist, min_ang_diff_hist, 
                            max_ang_diff_hist,heatmaps,px_u_hist, px_v_hist))
input_config = "all_info"

errors: np.ndarray = errors.reshape((num_points * num_angles, errors.shape[2]))
e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
e_rot = np.array([errors[:,3]]).T

errors = np.hstack((e_trans, e_rot))
trans_errors = errors[:,0] 

max_error = 5.0
trans_errors = np.clip(trans_errors, None, max_error)

trans_errors = standardize(trans_errors, max_error)
trans_errors = np.array([trans_errors]).T
print(np.mean(trans_errors))
print(np.std(trans_errors))


dataset = RegressionDataset(histogram_data, trans_errors)



# use 10% of training data for validation
train_set_size = int(len(dataset) * 0.9)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, 8, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=0)

# for batch, data in enumerate(train_loader):
#     features, label = data

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="acc", mode="min",save_weights_only=False)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"Regression/{input_config}")
model = ViewpointRegressor(histogram_data.shape[1], max_error=max_error)
trainer = pl.Trainer(max_epochs=500, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
