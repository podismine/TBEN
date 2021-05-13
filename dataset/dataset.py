#coding:utf8
import pandas as pd
from torch.utils import data
import numpy as np
from sklearn.utils import shuffle
import nibabel as nib

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from utils.data import *

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class MultiBranch_Data(data.Dataset):

    def __init__(self,csv,train=True):
        
        self.train = train

        df = pd.read_csv(csv)

        self.imgs = list(df['t1'])
        self.lbls = [float(f) for f in list(df['age'])]
        self.sexs = [float(f) for f in list(df['sex'])]
        self.dataset_len = len(self.imgs)

        
    def __getitem__(self,index):

        img = self.imgs[index]
        lbl = self.lbls[index]

        y, bc = generate_label(lbl,sigma = 2, bin_step = 1)

        img = img / np.mean(img)
        
        if self.train:
            img = coordinateTransformWrapper(img,maxDeg=20,maxShift=5, mirror_prob = 0)

        img = img[np.newaxis,...]
        return img, lbl, y, bc
    
    def __len__(self):
        return len(self.imgs)
