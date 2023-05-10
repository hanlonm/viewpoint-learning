from dataloader import ClassifierDataset
from torch.utils.data import DataLoader, random_split
from MLP_classifier import ViewpointClassifier
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import h5py


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
min_dist_hist = histogram_data[:,2:12]
max_dist_hist = histogram_data[:,12:22]
min_ang_hist =histogram_data[:,22:32]
max_ang_hist = histogram_data[:,32:42]
min_ang_diff_hist =histogram_data[:,42:52]
max_ang_diff_hist =histogram_data[:,52:62]
heatmaps =histogram_data[:,62:126]
px_u_hist=histogram_data[:,126:136]
px_v_hist=histogram_data[:,136:146]



# histogram_data = np.hstack((ranges,min_dist_hist, max_dist_hist, min_ang_hist, max_ang_hist, min_ang_diff_hist, 
#                             max_ang_diff_hist,heatmaps,px_u_hist, px_v_hist))
input_config = "only_mapping_info"
histogram_data = np.hstack((min_dist_hist, max_dist_hist, min_ang_hist, max_ang_hist,heatmaps,px_u_hist, px_v_hist))
errors: np.ndarray = errors.reshape((num_points * num_angles, errors.shape[2]))
e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
e_rot = np.array([errors[:,3]]).T

errors = np.hstack((e_trans, e_rot))

labels = np.logical_and(errors[:, 0] < 0.05, errors[:, 1] < 0.5)
labels = labels.astype(int)
labels = np.array([labels]).T

pos = np.argwhere(labels==1)[:,0]
neg = np.argwhere(labels==0)[:,0]
pos = np.random.choice(pos, 1000)
neg = np.random.choice(neg, 1000)

pos_hist = histogram_data[pos]
neg_hist = histogram_data[neg]
pos_labels = labels[pos]
neg_labels = labels[neg]

histogram_data = np.vstack((pos_hist, neg_hist))
labels = np.vstack((pos_labels,neg_labels))

tot = np.sum(labels)

dataset = ClassifierDataset(histogram_data, labels)



# use 10% of training data for validation
train_set_size = int(len(dataset) * 0.9)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, 8, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_set, 8, shuffle=False, num_workers=0)

for batch, data in enumerate(train_loader):
    features, label = data

checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="acc", mode="max",save_weights_only=True)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"Classifier/{input_config}")
model = ViewpointClassifier(histogram_data.shape[1])
trainer = pl.Trainer(max_epochs=100, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)