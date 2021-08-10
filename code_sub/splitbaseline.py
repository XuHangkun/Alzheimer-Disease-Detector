import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from useless_ids import useless_ids
from collections import Iterable

class SplitBaselineConfig:
    def __init__(self,
            atlas_roi_path="atlas_roi.csv",
            atlas=["AAL"],
            ctype="all",
            out_dim = 3,
            dropout = 0.6
        ):
        if isinstance(atlas,Iterable):
            self.atlas = [item.lower() for item in atlas]
        else:
            if atlas.lower() == "all":
                self.atlas = "all"
            else:
                self.atlas = [atlas.lower()]
        self.ctype = ctype
        self.atlas_roi_path = atlas_roi_path
        self.out_dim = out_dim
        self.dropout = dropout

    def initialize(self):
        self.x_ids = self.cal_x_ids()
        self.in_dim = len(self.x_ids)
        self.n_hidden = min(self.in_dim // 8,512)
        self.n_hidden = max(self.n_hidden,32)

    def atlas_contain(self,atlas):
        """judge if atlas contains in atlas
        """
        if self.atlas == "all":
            return True
        else:
            for item in self.atlas:
                if item.lower() in atlas.lower():
                    return True
        return False


    def cal_x_ids(self):
        atlas_roi_df = pd.read_csv(self.atlas_roi_path)
        ctype = self.ctype.lower()
        x_ids = []
        if ctype == "gmv":
            for i in range(len(atlas_roi_df)):
                if "mesh" in atlas_roi_df["Atlas"][i] and self.atlas_contain(atlas_roi_df["Atlas"][i]):
                    x_ids.append(atlas_roi_df["dim"][i])

        elif ctype == "ct":
            for i in range(len(atlas_roi_df)):
                if "mesh" not in atlas_roi_df["Atlas"][i] and self.atlas_contain(atlas_roi_df["Atlas"][i]):
                    x_ids.append(atlas_roi_df["dim"][i])
        else:
            for i in range(len(atlas_roi_df)):
                if self.atlas_contain(atlas_roi_df["Atlas"][i]):
                    x_ids.append(atlas_roi_df["dim"][i])
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
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim,in_dim)
        )
        self.relu = nn.ELU()

    def forward(self,x):
        return self.relu(self.net(x) + x)

class SplitBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_ids = config.x_ids

        self.net = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.in_dim, self.config.n_hidden),
            nn.ELU(),
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
    config = SplitBaselineConfig(atlas=["Hammers"])
    config.initialize()
    print(config.__dict__)
    model = SplitBaseline(config)
    model.load_state_dict(torch.load("model.pth", map_location ='cpu'))
    model.eval()
    x = np.load("../data/train/Subject_0004.npy")
    x = np.nan_to_num(x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
    mean = np.load("mean.npy")
    std = np.load("std.npy")
    x = (x - mean)/std
    x = np.nan_to_num(x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
    x = torch.tensor(x).unsqueeze(0).float()
    print(x.shape)
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
