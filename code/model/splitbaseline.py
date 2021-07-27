import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .useless_ids import useless_ids

class SplitBaselineConfig:
    def __init__(self,
            atlas_roi_path="../data/atlas_roi.csv",
            atlas="all",
            ctype="all",
            out_dim = 3,
            dropout = 0.1
        ):
        self.atlas = atlas
        self.ctype = ctype
        self.atlas_roi_path = atlas_roi_path
        self.out_dim = out_dim
        self.dropout = dropout

    def initialize(self):
        self.x_ids = self.cal_x_ids()
        self.in_dim = len(self.x_ids)
        self.n_hidden = self.in_dim // 2

    def cal_x_ids(self):
        atlas_roi_df = pd.read_csv(self.atlas_roi_path)
        atlas = self.atlas.lower()
        ctype = self.ctype.lower()
        x_ids = []
        if atlas == "all":
            if ctype == "gmv":
                x_ids = np.array([atlas_roi_df["dim"][i] for i in range(len(atlas_roi_df)) if "mesh" in atlas_roi_df["Atlas"][i]])
            elif ctype == "ct":
                x_ids = np.array([atlas_roi_df["dim"][i] for i in range(len(atlas_roi_df)) if "mesh" not in atlas_roi_df["Atlas"][i]])
            else:
                x_ids = atlas_roi_df["dim"].to_numpy()
        else:
            if ctype == "gmv":
                for i in range(len(atlas_roi_df)):
                    if "mesh" in atlas_roi_df["Atlas"][i] and atlas in atlas_roi_df["Atlas"][i].lower():
                        x_ids.append(i)
            elif ctype == "ct":
                for i in range(len(atlas_roi_df)):
                    if "mesh" not in atlas_roi_df["Atlas"][i] and atlas in atlas_roi_df["Atlas"][i].lower():
                        x_ids.append(i)
            else:
                for i in range(len(atlas_roi_df)):
                    if atlas in atlas_roi_df["Atlas"][i].lower():
                        x_ids.append(i)
        # filter useless id here
        new_ids = []
        for ids in x_ids:
            if ids in useless_ids:
                continue
            else:
                new_ids.append(ids)
        return np.array(new_ids)

class FullResLayer(nn.Module):
    def __init__(self,in_dim,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim,in_dim)
        )
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        return self.relu(self.net(x) + x)

class SplitBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_ids = config.x_ids

        self.net = nn.Sequential(
            nn.Linear(self.config.in_dim, self.config.n_hidden),
            nn.LeakyReLU(),
            FullResLayer(self.config.n_hidden,self.config.dropout),
            nn.Dropout(self.config.dropout),
            FullResLayer(self.config.n_hidden,self.config.dropout),
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
    config = SplitBaselineConfig(atlas="Brodmann")
    config.initialize()
    print(config.__dict__)
    model = SplitBaseline(config)
    x = torch.randn((32,15000))
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
