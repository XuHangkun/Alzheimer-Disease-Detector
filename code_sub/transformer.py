import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from useless_ids import useless_ids
from collections import Iterable

class TransformerConfig:
    def __init__(self,
            atlas_roi_path="../data/atlas_roi.csv",
            n_layers = 24,
            hidden_size = 256,
            out_dim = 3,
            dropout = 0.1,
            n_heads = 8,
            device = "cpu"
        ):
        self.atlas_roi_path = atlas_roi_path
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.device = device

    def initialize(self):
        self.x_ids = self.cal_x_ids()

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
        atlases = []
        x_ids = {}
        for atlas in list(atlas_roi_df["Atlas"]):
            if atlas not in atlases:
                atlases.append(atlas)
                x_ids[atlas] = []

        for i in range(len(atlas_roi_df)):
            atlas = atlas_roi_df["Atlas"][i]
            dim = atlas_roi_df["dim"][i]
            # filter useless id here
            if  dim in useless_ids:
                continue
            else:
                x_ids[atlas].append(dim)
        for key in x_ids.keys():
            x_ids[key] = np.array(x_ids[key])
        new_xids = []
        for atlas in atlases:
            new_xids.append(x_ids[atlas])
        return new_xids

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

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_ids = config.x_ids
        self.device = config.device
        self.projs = [nn.Linear(len(item),self.config.hidden_size).to(self.device) for item in self.x_ids]
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=self.config.n_heads),
                num_layers = self.config.n_layers
                )

        self.fcs = [
            nn.Sequential(nn.Linear(self.config.hidden_size,self.config.out_dim),nn.Softmax(-1)).to(self.device)
            for item in self.x_ids
        ]

    def _preprocess(self,x):
        if len(x.shape) == 1:
            return x[self.x_ids]
        if len(x.shape) == 2:
            return x[:,self.x_ids]

    def forward(self, x):
        """
        args:
            x : input of feaure [batchsize,feature_len]
        return:
            x : possibility of three class
        Shape:
            input : [batch_size, in_dim]
            output : [batch_size,out_dim]
        """
        features = []
        for ids,proj in zip(self.x_ids,self.projs):
            features.append(proj(x[:,ids]))
        features = torch.stack(features,dim=0).to(self.device) # [feature num,batch size,hidden size]
        features = self.encoder(features)
        outs = []
        for i in range(features.size(0)):
            a_feature = features[i]
            outs.append(self.fcs[i](a_feature))
        outs = torch.stack(outs,dim=0).to(self.device) # [feature num,batch size,out_dim]
        if self.training:
            return outs
        else:
            return torch.mean(outs,dim=0)

def test():
    config = TransformerConfig()
    config.initialize()
    print(config.__dict__)
    model = Transformer(config)
    model.load_state_dict(torch.load("model.pth", map_location ='cpu'))
    model.eval()
    x = np.load("../data/train/Subject_0006.npy")
    mean = np.load("mean.npy")
    std = np.load("std.npy")
    x = (x - mean)/std
    x = torch.tensor(x).unsqueeze(0).float()
    print(x.shape)
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
