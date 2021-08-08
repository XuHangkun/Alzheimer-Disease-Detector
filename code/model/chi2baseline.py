import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .useless_ids import useless_ids
from collections import Iterable

class Chi2BaselineConfig:
    def __init__(self,
            chi2_info_path="../data/chi2_info.csv",
            chi2_cut = 0.5,
            out_dim = 3,
            dropout = 0.1
        ):
        self.chi2_info_path = chi2_info_path
        self.chi2_cut = chi2_cut
        self.out_dim = out_dim
        self.dropout = dropout

    def initialize(self):
        self.x_ids = self.cal_x_ids()
        self.in_dim = len(self.x_ids)
        self.n_hidden = min(self.in_dim // 2,512)

    def cal_x_ids(self):
        chi2_info = pd.read_csv(self.chi2_info_path)
        chi2_info = chi2_info[chi2_info["chi2"] > self.chi2_cut]
        dims = []
        chi2s = []
        for i in range(len(chi2_info)):
            chi2 = "%.7f"%(chi2_info["chi2"][i])
            if chi2 not in chi2s and chi2_info["dim"][i] not in useless_ids:
                chi2s.append(chi2)
                dims.append(chi2_info["dim"][i])
        return np.array(dims)

class FullResLayer(nn.Module):
    def __init__(self,in_dim,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim,in_dim)
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.net(x) + x)

class Chi2Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_ids = config.x_ids

        self.net = nn.Sequential(
            nn.Dropout(0.65),
            nn.Linear(self.config.in_dim, self.config.n_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            FullResLayer(self.config.n_hidden,self.config.dropout),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_hidden, self.config.out_dim),
            nn.Softmax(dim=-1)
        )

    def _preprocess(self,x):
        if len(x.shape) == 1:
            return x[self.x_ids]
        if len(x.shape) == 2:
            return x[:,self.x_ids]

    def forward(self, x):
        """
        args:
            x : input of feaure
        return:
            x : possibility of three class
        Shape:
            input : [batch_size, in_dim]
            output : [batch_size,out_dim]
        """
        x = self._preprocess(x)
        x = self.net(x)
        return x

def test():
    config = Chi2BaselineConfig()
    config.initialize()
    print(config.__dict__)
    model = Chi2Baseline(config)
    x = torch.randn((32,28169))
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
