from dataloader import ViewpointDataset
from torch.utils.data import DataLoader, random_split
from viewpoint_encoder import ViewpointAutoEncoder
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
labels = hf["errors"][:]
histogram_data = histogram_data.reshape(
    (num_points * num_angles, histogram_data.shape[2]))
labels: np.ndarray = labels.reshape((num_points * num_angles, labels.shape[2]))
e_trans = np.array([np.linalg.norm(labels[:,:3], axis=1)]).T
e_rot = np.array([labels[:,3]]).T

labels = np.hstack((e_trans, e_rot))

dataset = ViewpointDataset(histogram_data, labels)

auto_encoder = ViewpointAutoEncoder.load_from_checkpoint("/local/home/hanlonm/viewpoint-learning/AutoEncoder/lightning_logs/version_0/checkpoints/epoch=95-step=21600.ckpt", 
                                                         input_dim=histogram_data.shape[1], encoding_dim=64)
encoder = auto_encoder.encoder
encoder.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
encoder.to(device)

test_loader = DataLoader(dataset)
encodings = []
for batch, data in enumerate(test_loader):
    features, label = data
    features = features.to(device)
    preds = encoder(features)
    encodings.append(preds.tolist())
encodings = np.array(encodings)

encodings = encodings.reshape((num_points,num_angles,-1))
hf.create_dataset("encodings", data=encodings)
print()