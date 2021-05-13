import torch.nn as nn
import torch
def my_KLDivLoss(x, y):
    loss = nn.KLDivLoss(reduction='sum')(x, y + 1e-16) / y.shape[0]
    return loss
