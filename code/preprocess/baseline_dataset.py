import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import torch.utils.data as data

class BaselineDataset(data.Dataset):
    def __init__(self, x, y ,n_aug = 0 ,tuning=0.015):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)
        self.tuning = tuning
        self.n_aug = n_aug
        self.ratios = [1./(1 + 1*n_aug),(1.0 + n_aug)/(1 + 1*n_aug)]

    def __getitem__(self, index):
        rand = np.random.random()
        if index < self.ratios[0]*len(self):
            index = index%(len(self.y))
            xi = self.x[index]
        elif index < self.ratios[1]*len(self):
            index = index%(len(self.y))
            for i in range(10):
                o_ratio = self.tuning*abs(np.random.randn())
                o_index = int(np.random.random()*len(self.y))
                if self.y[index] == self.y[o_index]:
                    o_ratio *=5
                    break
            xi = self.x[index]*(1 - o_ratio) + o_ratio*self.x[o_index]
        else:
            index = index%(len(self.y))
            xi = self.x[index] * (1 + self.tuning*np.random.randn(len(self.x[index])))
        xi = xi.float()
        yi = self.y[index]
        return xi, yi

    def __len__(self):
        return len(self.y)*(1+1*self.n_aug)
