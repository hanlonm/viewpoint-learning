import torch
import os
import h5py
from MLP_classifier import ViewpointClassifier
import numpy as np
from utils import normalize, standardize, create_dataset


model = ViewpointClassifier.load_from_checkpoint(
                    "/local/home/hanlonm/mt-matthew/data/models/classifiers/230628/best_test.ckpt",
                    input_dim=146)
model.eval()


home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")


hf = h5py.File(str(home_dir)+"/mt-matthew/data/training_data/230627_hist.h5", "r")
train_environments = ["00195", "00654", "00111"]
train_histograms, train_trans_errors, train_rot_errors = create_dataset(hf, train_environments, 5.0)



# Generate a random input vector
input_vector = torch.tensor(train_histograms[10568], dtype=torch.float32).cuda()
# input_vector = torch.randn(146).cuda()

# Enable gradients for the input vector
input_vector.requires_grad = True

# Forward pass
output = model(input_vector.unsqueeze(0))

# Create a tensor for gradient calculation
grad_output = torch.zeros_like(output)
target_class = 1
grad_output[:, target_class] = 1.0  # Set the target class index for gradient calculation

# Calculate gradients
model.zero_grad()
output.backward(gradient=grad_output)

# Calculate integrated gradients
baseline = torch.zeros_like(input_vector)
num_steps = 10000
alphas = torch.linspace(0, 1, num_steps)
interpolated_inputs = [
    baseline + alpha * (input_vector - baseline.detach()) for alpha in alphas
]
interpolated_inputs = [x.requires_grad_(True) for x in interpolated_inputs]

interpolated_outputs = torch.stack([model(x.unsqueeze(0)) for x in interpolated_inputs])
grads = torch.autograd.grad(
    outputs=interpolated_outputs,
    inputs=interpolated_inputs,
    grad_outputs=torch.ones_like(interpolated_outputs),
    create_graph=True,
    retain_graph=True,
)[0]

integrated_gradients = torch.mean(grads, dim=0) * (input_vector - baseline)

# Normalize the integrated gradients
integrated_gradients = (integrated_gradients - integrated_gradients.min()) / (
    integrated_gradients.max() - integrated_gradients.min()
)

# Convert the integrated gradients to a numpy array for visualization
integrated_gradients = integrated_gradients.detach().cpu().numpy()

# Visualize the integrated gradients
import matplotlib.pyplot as plt

labels = 2 * ["ranges"] + 10*["min_dist"]+10*["max_dist"]+10*["min_ang"]+10*["max_ang"]+10*["min_ang_diff"]+10*["max_ang_diff"]+64*["heat"]+10*["pxu"]+10*["pxv"]
indices = np.argsort(integrated_gradients)[-25:]
for idx in reversed(indices):
    print(labels[idx] + f" {integrated_gradients[idx]}")
plt.bar(range(len(input_vector)), integrated_gradients)
plt.xlabel("Input Features")
plt.ylabel("Importance")
plt.show()