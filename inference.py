from models.TBEN import TBEN
import torch
import nibabel as nib
import numpy as np
import os

model = TBEN(depth = 8, dim = 128, mlp_dim=512, pool = 'cls', tr_drop = 0.5)
checkpoint = torch.load('checkpoints/TBEN_checkpoints_best2', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

labelset = np.array([i + 14 for i in range(84)])
bc_tensor = torch.tensor(labelset).float()

test_obj = np.load("test_data.npz")
all_test_data = test_obj['x']
labels = test_obj['y']

loss = 0
for i in range(len(labels)):
    data = all_test_data[i,...]
    test_tensor = torch.tensor(data[np.newaxis,np.newaxis,...]).float()
    out = model(test_tensor)
    prob = torch.exp(out)
    pred = torch.sum(prob * bc_tensor, dim = 1).cpu().detach().numpy()[0,...]
    print(abs(pred - labels[i]), pred, labels[i])
    loss += abs(pred - labels[i])
print(loss/len(labels))