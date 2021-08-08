import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .useless_ids import useless_ids
from collections import Iterable

class FBaselineConfig:
    def __init__(self,
            anova_info_path="../data/anova_info.csv",
            f_cut = 150,
            out_dim = 3,
            dropout = 0.1
        ):
        self.anova_info_path = anova_info_path
        self.f_cut = f_cut
        self.out_dim = out_dim
        self.dropout = dropout

    def initialize(self):
        self.x_ids = self.cal_x_ids()
        self.in_dim = len(self.x_ids)
        self.n_hidden = min(self.in_dim // 6,512)

    def cal_x_ids(self):
        anova_info = pd.read_csv(self.anova_info_path)
        anova_info = anova_info[anova_info["f"] > self.f_cut]
        dims = []
        fs = []
        for i in range(len(anova_info)):
            f = "%.7f"%(anova_info["f"][i])
            if f not in fs and anova_info["dim"][i] not in useless_ids:
                fs.append(f)
                dims.append(anova_info["dim"][i])
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

class FBaseline(nn.Module):
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
            FullResLayer(self.config.n_hidden,self.config.dropout),
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
    config = FBaselineConfig()
    config.initialize()
    print(config.__dict__)
    model = FBaseline(config)
    x = torch.randn((32,28169))
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
