from models.TBEN import TBEN
from models.sfcn import SFCN
import torch
import nibabel as nib
import numpy as np
import os
import argparse
from utils.data import dataaug, num2vect

def infer_model(model, data, bc):
    test_tensor = torch.tensor(data[np.newaxis,np.newaxis,...]).float()
    out = model(test_tensor)
    prob = torch.exp(out)
    pred = torch.sum(prob * bc, dim = 1).cpu().detach().numpy()[0,...]
    return pred


predicts = []
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./', help='path to dataset')
parser.add_argument('--age', default=-1,type=float,help='brain age')
args = parser.parse_args()
############################################################################################## 
# load data
print("Loading Data...")
try:
    data = nib.load(args.data).get_fdata()
    data /= np.mean(data)
except:
    print("Fail to get data, please check the data format. Only nii/nii.gz is support.")
    exit()
labelset = np.array([i + 14 for i in range(84)])
bc_tensor84 = torch.tensor(labelset).float()

bc_tensor40 = torch.tensor(num2vect(bin_range = [14, 94], bin_step = 2, sigma = 1)).float()
bc_tensor80 = torch.tensor(num2vect(bin_range = [14, 94], bin_step = 1, sigma = 1)).float()
# load model
print("Loading Model...")

############################################################################################## 
print("Loading First Model(TBEN)...")
model_tben = TBEN(depth = 8, dim = 128, mlp_dim=512, pool = 'cls', tr_drop = 0.5)
checkpoint = torch.load('checkpoints/TBEN_checkpoints_best2', map_location='cpu')
model_tben.load_state_dict(checkpoint['state_dict'])

predicts.append(infer_model(model_tben, data, bc_tensor84))
for i in range(8):
    predicts.append(infer_model(model_tben, dataaug(data), bc_tensor84))

pred1 = np.mean(predicts)

############################################################################################## 
print("Laoding Second Model(SFCN)...")
model_sfcn40 = torch.load("checkpoints/task2_sfcn_40_3.24", map_location='cpu')
model_sfcn80 = torch.load("checkpoints/task2_sfcn_80_3.32", map_location='cpu')

predicts.append(infer_model(model_sfcn40, data, bc_tensor40))
for i in range(2):
    predicts.append(infer_model(model_sfcn40, dataaug(data), bc_tensor40))

predicts.append(infer_model(model_sfcn80, data, bc_tensor80))
for i in range(1):
    predicts.append(infer_model(model_sfcn80, dataaug(data), bc_tensor80))

pred2 = np.mean(predicts)

if args.age == -1:
    mean_age = np.mean([pred1,pred2])
    min_age = max(min(pred1,pred2), mean_age - 10)
    max_age = min(max(pred1, pred2), mean_age + 10)
    print("Cause You have not provided your chonological age, we guessed your Brain age is %s, between %s and %s" % (mean_age, min_age, max_age))
else:
    pred = sorted([v - args.age for v in predicts])[0] + args.age
    if abs(pred - args.age) >= 10:
        pred = args.age + np.sign(pred - args.age) * (1 - np.exp(0 - abs(pred - args.age))) * 10
    print("Your Brain age is %s, " % (pred))


