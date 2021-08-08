import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import torch.utils.data as data
from sklearn.decomposition import PCA

class PCADataset(data.Dataset):
    def __init__(self, x, y ,pca = None, n_comp=20 ,sel_dims = None,
        n_aug = 0, tuning=0.015):
        self.x = x
        self.y = torch.from_numpy(y)
        if pca == None:
            self.pca = PCA(n_components = n_comp)
        else:
            self.pca = pca
        self.model_in_dim = self.pca.n_components_
        if not sel_dims:
            self.sel_dims = np.array(range(self.x.shape[1]))
        else:
            self.sel_dims = sel_dims
        self.preprocess()
        self.tuning = tuning
        self.n_aug = n_aug
        self.ratios = [1./(1 + 2*n_aug),(1.0 + n_aug)/(1 + 2*n_aug), (1.0 + 2.0*n_aug)/(1 + 2*n_aug)]

    def preprocess(self):
        new_x = self.x[:,self.sel_dims]
        self.x = self.pca.transform(new_x)
        # normalization
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.x = (self.x - self.mean)/self.std
        self.x = torch.from_numpy(self.x).float()

    def __getitem__(self, index):
        rand = np.random.random()
        if index < self.ratios[0]*len(self):
            index = index%(len(self.y))
            xi = self.x[index]
        elif index < self.ratios[1]*len(self):
            index = index%(len(self.y))
            o_ratio = self.tuning*abs(np.random.randn())
            o_index = int(np.random.random()*len(self.y))
            if self.y[index] == self.y[o_index]:
                o_ratio *=2
            xi = self.x[index]*(1 - o_ratio) + o_ratio*self.x[o_index]
        else:
            index = index%(len(self.y))
            xi = self.x[index] * (1 + self.tuning*np.random.randn(len(self.x[index])))
        xi = xi.float()
        yi = self.y[index]
        return xi, yi

    def __len__(self):
        return len(self.y)*(1+2*self.n_aug)
