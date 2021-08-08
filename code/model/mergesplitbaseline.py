import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .useless_ids import useless_ids
from collections import Iterable
from .splitbaseline import SplitBaselineConfig,SplitBaseline

class MergeSplitBaselineConfig:
    def __init__(self,
            atlas_roi_path="../data/atlas_roi.csv",
            atlas="all",
            ctype="all",
            out_dim = 3,
            dropout = 0.6,
            folds = 10
        ):
        self.splitbaselineconfig = SplitBaselineConfig(
            atlas_roi_path=atlas_roi_path,
            atlas=atlas,
            ctype=ctype,
            out_dim = out_dim,
            dropout = dropout
        )

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
    
    def forward(self,x):
        xs = []
        xs.append(self.net_fold1(x))
        xs.append(self.net_fold2(x))
        xs.append(self.net_fold3(x))
        xs.append(self.net_fold4(x))
        xs.append(self.net_fold5(x))
        xs.append(self.net_fold6(x))
        xs.append(self.net_fold7(x))
        xs.append(self.net_fold8(x))
        xs.append(self.net_fold9(x))
        xs.append(self.net_fold10(x))
        x = torch.stack(xs)
        x = torch.mean(x,axis=0)
        return x
        
def merge_models():
    best_epoches = [261,417,457,449,496,468,309,285,370,490]
    paths = [
        "../model/baseline/baseline_AAL_lr1.e-3_dp0.6_fold%d_epoch%d.pth model.pth"%(x,y) for x,y in zip(range(1,11),best_epoches)
    ]
    config = MergeSplitBaselineConfig()
    model = MergeSplitBaseline(config)
    model.load_models(paths)
    torch.save(model.state_dict(), "model.pth")
    # do a simple test here

def test():
    pass

if __name__ == "__main__":
    merge_models()
    test()