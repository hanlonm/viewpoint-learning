from dataloader import ClassifierDataset, PCTTransformerDataset, pct_transformer_collate, MyDataModule, TestCallback
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from viewpoint_pct import PCTViewpointTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import os
from utils import normalize, standardize, pre_process, remove_nan_rows, create_transformer_dataset, create_variable_transformer_dataset


home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")

# hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/token_test_5_var.h5", "r+")
# hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/test_new_envs_1.h5", "r+")
#hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/230522_100.h5", "r+")
# hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/0612_100_new_envs.h5", "r+")
#hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/100_50_230629.h5", "r")
#hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/100_50_230624.h5", "r")
# hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/100_50_230704_dino_env_12.h5", "r")
hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/opt_occ_100_50_230724.h5", "r")


print(hf.keys())
# num_points = hf.attrs["num_points"]
# num_angles = hf.attrs["num_angles"]

input_config = "dino_3_10-5_16_occ_opt_norm_small_noheat"

train_environments = ["00269_opt", "00067_opt", "00596_opt", "00638_opt", "00700_opt"]
# train_environments = ["00067"]
# train_environments = ["00067", "00596", "00638", "00700"]
test_environments = ["00195_opt", "00654_opt", "00111_opt", "00403_opt"]
# train_environments = ["00067", "00596", "00638", "00700", "00654"]
# test_environments = ["00195"]

max_error = 5.0
# train_tokens, train_trans_errors, train_rot_errors = create_transformer_dataset(hf, train_environments, max_error)
# test_tokens, test_trans_errors, test_rot_errors = create_transformer_dataset(hf, test_environments, max_error)
train_tokens, train_trans_errors, train_rot_errors = create_variable_transformer_dataset(hf, train_environments, max_error)
test_tokens, test_trans_errors, test_rot_errors = create_variable_transformer_dataset(hf, test_environments, max_error)

hf.close()

train_errors = np.hstack((train_trans_errors, train_rot_errors))
train_labels = np.logical_and((train_errors[:, 0]*max_error) < 0.10, train_errors[:, 1] < 1)
train_labels = train_labels.astype(int)
train_labels = np.array([train_labels]).T

test_errors = np.hstack((test_trans_errors, test_rot_errors))
test_labels = np.logical_and((test_errors[:, 0]*max_error) < 0.10, test_errors[:, 1] < 1)
test_labels = test_labels.astype(int)
test_labels = np.array([test_labels]).T

pos = np.argwhere(train_labels==1)[:,0]
neg = np.argwhere(train_labels==0)[:,0]
print(len(pos))
print(len(neg))
pos = np.random.choice(pos, 12500)
neg = np.random.choice(neg, 12500)

test_pos = np.argwhere(test_labels==1)[:,0]
test_neg = np.argwhere(test_labels==0)[:,0]
print(len(test_pos))
print(len(test_neg))
test_pos = np.random.choice(test_pos, 10000)
test_neg = np.random.choice(test_neg, 10000)
test_pos_toks = test_tokens[test_pos]
test_neg_toks = test_tokens[test_neg]
test_pos_labels = test_labels[test_pos]
test_neg_labels = test_labels[test_neg]
test_tokens = np.concatenate((test_pos_toks, test_neg_toks))
test_labels = np.vstack((test_pos_labels, test_neg_labels))


pos_toks = train_tokens[pos]
neg_toks = train_tokens[neg]
pos_labels = train_labels[pos]
neg_labels = train_labels[neg]

# input_data = np.vstack((pos_toks, neg_toks))
input_data = np.concatenate((pos_toks, neg_toks))
train_labels = np.vstack((pos_labels, neg_labels))

tot = np.sum(train_labels)
print(np.sum(test_labels)/len(test_labels))

train_dataset = PCTTransformerDataset(input_data, train_labels)
test_dataset = PCTTransformerDataset(test_tokens, test_labels)

# use 10% of training data for validation
train_set_size = int(len(train_dataset) *0.8)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, 16, True, num_workers=8, collate_fn=pct_transformer_collate, pin_memory=True)
val_loader = DataLoader(valid_set, 32, False, num_workers=8, collate_fn=pct_transformer_collate, pin_memory=True)
test_loader = DataLoader(test_dataset, 32, False, num_workers=8, collate_fn=pct_transformer_collate, pin_memory=True)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc/dataloader_idx_0", mode="max",save_weights_only=True)
checkpoint_test_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc/dataloader_idx_1", mode="max",save_weights_only=True,filename=f'best_test_{input_config}')
test_cb = TestCallback(test_loader)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"PCT/{input_config}")
model = PCTViewpointTransformer()
trainer = pl.Trainer(max_epochs=75, logger=tb_logger, callbacks=[checkpoint_callback, LearningRateMonitor("step"), checkpoint_test_callback], profiler="simple")
trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
trainer.fit(model,train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader])
trainer.test(dataloaders=test_loader, ckpt_path="best")
