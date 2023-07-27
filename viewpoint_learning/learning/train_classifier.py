import os
from dataloader import ClassifierDataset
from torch.utils.data import DataLoader, random_split
from MLP_classifier import ViewpointClassifier
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py
import matplotlib.pyplot as plt
from utils import normalize, standardize, create_dataset

home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")


# hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/0601_100_occ.h5", "r+")
# hf = h5py.File("/local/home/hanlonm/mt-matthew/data/training_data/230522_100.h5", "r+")
# hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/230627_hist.h5", "r")
# hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/100_50_histograms.h5", "r")
#hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/100_50_230705_dino_3_hist.h5", "r")
hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/opt_occ_100_50_230724.h5", "r")



print(hf.keys())
# num_points = hf.attrs["num_points"]
# num_angles = hf.attrs["num_angles"]

input_config = "mlp_opt_occ_1-5_32"

train_environments = ["00067_opt", "00596_opt", "00638_opt", "00700_opt", "00269_opt"]
test_environments = ["00195_opt", "00654_opt", "00111_opt", "00403_opt"]
# test_environments = ["00195", "00654", "00111", "00403"]
# train_environments = ["00111", "00269", "00403", "00067", "00596", "00638", "00700"]
# test_environments = ["00195", "00654"]


max_error = 5.0
train_histograms, train_trans_errors, train_rot_errors = create_dataset(hf, train_environments, max_error)
test_histograms, test_trans_errors, test_rot_errors = create_dataset(hf, test_environments, max_error)

train_errors = np.hstack((train_trans_errors, train_rot_errors))
train_labels = np.logical_and((train_errors[:, 0]*max_error) < 0.10, train_errors[:, 1] < 1)
train_labels = train_labels.astype(int)
train_labels = np.array([train_labels]).T

test_errors = np.hstack((test_trans_errors, test_rot_errors))
test_labels = np.logical_and((test_errors[:, 0]*max_error) < 0.10, test_errors[:, 1] < 1)
test_labels = test_labels.astype(int)
test_labels = np.array([test_labels]).T
test_pos = np.argwhere(test_labels==1)[:,0]
test_neg = np.argwhere(test_labels==0)[:,0]
test_pos = np.random.choice(test_pos, 10000)
test_neg = np.random.choice(test_neg, 10000)
test_pos_histograms = test_histograms[test_pos]
test_neg_histograms = test_histograms[test_neg]
test_pos_labels = test_labels[test_pos]
test_neg_labels = test_labels[test_neg]
test_histograms = np.vstack((test_pos_histograms, test_neg_histograms))
test_labels = np.vstack((test_pos_labels, test_neg_labels))

pos = np.argwhere(train_labels==1)[:,0]
neg = np.argwhere(train_labels==0)[:,0]
pos = np.random.choice(pos, 12500)
neg = np.random.choice(neg, 12500)

pos_hist = train_histograms[pos]
neg_hist = train_histograms[neg]
pos_labels = train_labels[pos]
neg_labels = train_labels[neg]

histogram_data = np.vstack((pos_hist, neg_hist))
train_labels = np.vstack((pos_labels,neg_labels))

tot = np.sum(train_labels)

train_dataset = ClassifierDataset(histogram_data, train_labels)
test_dataset = ClassifierDataset(test_histograms, test_labels)

print(np.sum(test_labels)/len(test_labels))



# use 20% of training data for validation
train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size])

train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=0)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc/dataloader_idx_0", mode="max",save_weights_only=True)
checkpoint_test_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc/dataloader_idx_1", mode="max",save_weights_only=True,filename='best_test')

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"Classifier/{input_config}")
model = ViewpointClassifier(histogram_data.shape[1])
layer_index = 0  # Index of the layer you want to access
layer_weights = model.model[layer_index].weight.flatten().tolist()
# plt.bar(range(len(layer_weights)), layer_weights)
# plt.show()
# print(model.model[2].weight.flatten().tolist())
trainer = pl.Trainer(max_epochs=100, logger=tb_logger, callbacks=[checkpoint_callback,checkpoint_test_callback])
trainer.fit(model,train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader])
trainer.test(dataloaders=test_loader, ckpt_path="best")
layer_index = 0  # Index of the layer you want to access
layer_weights = model.model[layer_index].weight.flatten().tolist()
# plt.bar(range(len(layer_weights)), layer_weights)
# print(model.model[2].weight.flatten().tolist())
# plt.show(block=True)

