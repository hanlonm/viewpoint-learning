from dataloader import RegressionDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from transformer import ViTReg
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


from utils import normalize, standardize, pre_process, remove_nan_rows, create_transformer_dataset

hf = h5py.File("/local/home/hanlonm/active-viewpoint-selection/data/training_data/token_test_4_100.h5", "r+")
#hf = h5py.File("/local/home/hanlonm/active-viewpoint-selection/data/training_data/230522_100.h5", "r+")
print(hf.keys())
num_points = hf.attrs["num_points"]
num_angles = hf.attrs["num_angles"]

input_config = "token_1024"

train_environments = ["00067", "00596", "00638", "00700"]
test_environments = ["00195", "00654"]
# train_environments = ["00067", "00596", "00638", "00700", "00654"]
# test_environments = ["00195"]

max_error = 5.0
train_tokens, train_trans_errors, train_rot_errors = create_transformer_dataset(hf, train_environments, max_error)
test_tokens, test_trans_errors, test_rot_errors = create_transformer_dataset(hf, test_environments, max_error)


# Compute the histogram
hist, bin_edges = np.histogram(train_trans_errors, bins=100)
non_empty_bins = hist > 0
hist = hist[non_empty_bins]
bin_edges = bin_edges[:-1][non_empty_bins] 
num_bins = len(bin_edges)-1

samples_per_bin = int(train_trans_errors.shape[0]/num_bins)

# Assign each sample to its corresponding bin
bin_indices = np.digitize(train_trans_errors, bin_edges, True).flatten()
# Create a dictionary to store samples for each bin
bin_samples = {bin_num: [] for bin_num in range(1, len(bin_edges))}

idxs = []
for i in range(1, len(bin_edges)):
    idx = np.argwhere(bin_indices == i)[:,0]
    # if idx.size < 1:
    #     continue
    idx = np.random.choice(idx, samples_per_bin)
    idxs.append(idx)

idxs = np.concatenate(idxs)
_ = plt.hist(train_trans_errors[idxs], bins=num_bins)
plt.show()
train_tokens = train_tokens[idxs]
train_trans_errors = train_trans_errors[idxs]


train_dataset = RegressionDataset(train_tokens, train_trans_errors)
test_dataset = RegressionDataset(test_tokens, test_trans_errors)

# use 10% of training data for validation
train_set_size = int(len(train_dataset) * 0.9)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

val_test_set = ConcatDataset([valid_set, test_dataset])

train_loader = DataLoader(train_set, 8, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, 8, shuffle=False, num_workers=0)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="min",save_weights_only=True)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"TransformerReg/{input_config}")
model = ViTReg(model_kwargs={
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_heads": 2,
        "num_layers": 4,
        "dropout": 0.2,
    },
    lr=3e-4)
trainer = pl.Trainer(max_epochs=100, logger=tb_logger, callbacks=[checkpoint_callback, LearningRateMonitor("epoch")])
trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
trainer.fit(model, train_loader, val_loader)
trainer.test(dataloaders=test_loader, ckpt_path="best")
