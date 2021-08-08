import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Iterable
from splitbaseline import SplitBaselineConfig,SplitBaseline

class MergeSplitBaselineConfig:
    def __init__(self,
            atlas_roi_path="atlas_roi.csv",
            atlas=["Hammers"],
            ctype="all",
            out_dim = 3,
            dropout = 0.6,
            folds = 10
        ):
        self.atlas_roi_path=atlas_roi_path
        self.atlas=atlas
        self.ctype=ctype
        self.out_dim = out_dim
        self.dropout = dropout
    def initialize(self):
        self.splitbaselineconfig = SplitBaselineConfig(
            atlas_roi_path=self.atlas_roi_path,
            atlas=self.atlas,
            ctype=self.ctype,
            out_dim = self.out_dim,
            dropout = self.dropout
        )
        self.splitbaselineconfig.initialize()

class MergeSplitBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net_fold1 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold2 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold3 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold4 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold5 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold6 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold7 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold8 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold9 = SplitBaseline(self.config.splitbaselineconfig)
        self.net_fold10 = SplitBaseline(self.config.splitbaselineconfig)

    def load_models(self,paths):
        self.net_fold1.load_state_dict(torch.load(paths[0], map_location='cpu'))
        self.net_fold2.load_state_dict(torch.load(paths[1], map_location='cpu'))
        self.net_fold3.load_state_dict(torch.load(paths[2], map_location='cpu'))
        self.net_fold4.load_state_dict(torch.load(paths[3], map_location='cpu'))
        self.net_fold5.load_state_dict(torch.load(paths[4], map_location='cpu'))
        self.net_fold6.load_state_dict(torch.load(paths[5], map_location='cpu'))
        self.net_fold7.load_state_dict(torch.load(paths[6], map_location='cpu'))
        self.net_fold8.load_state_dict(torch.load(paths[7], map_location='cpu'))
        self.net_fold9.load_state_dict(torch.load(paths[8], map_location='cpu'))
        self.net_fold10.load_state_dict(torch.load(paths[9], map_location='cpu'))

    def judge_class(self,res):
        max_v = torch.max(res)
        res = res > max_v*0.999999
        print(res)
        for i in range(3):
            if res[i]:
                return i

    def forward(self,x):
        # x : [batchsize,feature_len]
        xs = []
        xs.append(self.net_fold1(x).squeeze())
        xs.append(self.net_fold2(x).squeeze())
        xs.append(self.net_fold3(x).squeeze())
        xs.append(self.net_fold4(x).squeeze())
        xs.append(self.net_fold5(x).squeeze())
        xs.append(self.net_fold6(x).squeeze())
        xs.append(self.net_fold7(x).squeeze())
        xs.append(self.net_fold8(x).squeeze())
        xs.append(self.net_fold9(x).squeeze())
        xs.append(self.net_fold10(x).squeeze())
        x = torch.stack(xs)
        ids = range(1,10)
        x = x[ids,:]
        x = torch.mean(x,dim=0)
        return x

def merge_models():
    # best_epoches = [261,417,457,449,496,468,309,285,370,490]
    # best_epoches = [266,127,481,468,215,401,489,441,325,439]      # with label smooth
    best_epoches = [900,797,921,914,780,811,796,781,546,922]      # Hammers
    #best_epoches = [479 for i in range(10)]
    paths = [
        "../model/Hammers/baseline_Hammers_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(range(1,11),best_epoches)
    ]
    print(paths)
    config = MergeSplitBaselineConfig()
    config.initialize()
    model = MergeSplitBaseline(config)
    model.load_models(paths)
    torch.save(model.state_dict(), "model.pth")
    # do a simple test here

def test():
    config = MergeSplitBaselineConfig()
    config.initialize()
    print(config.__dict__)
    model = MergeSplitBaseline(config)
    model.load_state_dict(torch.load("model.pth", map_location ='cpu'))
    model.eval()
    x = np.load("../data/train/Subject_0001.npy")
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
    #merge_models()
    test()
