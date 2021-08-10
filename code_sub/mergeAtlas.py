import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Iterable
from splitbaseline import SplitBaselineConfig,SplitBaseline

class MergeAtlasConfig:
    def __init__(self,
            atlas_roi_path="atlas_roi.csv",
            atlas = ["AAL","Hammers","rBN","AICHA","HarvardOxford"],
            atlas_folds = [1,7,1,0,0],
            out_dim = 3,
            dropout = 0.6
        ):
        self.atlas_roi_path=atlas_roi_path
        self.atlas = atlas
        self.atlas_folds = atlas_folds
        self.out_dim = out_dim
        self.dropout = dropout

    def initialize(self):
        self.splitbaselineconfigs = []
        for atlas in self.atlas:
            splitbaselineconfig = SplitBaselineConfig(
                atlas_roi_path=self.atlas_roi_path,
                atlas=[atlas],
                out_dim = self.out_dim,
                dropout = self.dropout
            )
            splitbaselineconfig.initialize()
            self.splitbaselineconfigs.append(splitbaselineconfig)

class MergeAtlasBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # AAL Module list
        self.aal_nets = nn.ModuleList([SplitBaseline(self.config.splitbaselineconfigs[0]) for i in range(self.config.atlas_folds[0])])
        self.hammers_nets = nn.ModuleList([SplitBaseline(self.config.splitbaselineconfigs[1]) for i in range(self.config.atlas_folds[1])])
        self.rbn_nets = nn.ModuleList([SplitBaseline(self.config.splitbaselineconfigs[2]) for i in range(self.config.atlas_folds[2])])
        self.aicha_nets = nn.ModuleList([SplitBaseline(self.config.splitbaselineconfigs[3]) for i in range(self.config.atlas_folds[3])])
        self.harvard_nets = nn.ModuleList([SplitBaseline(self.config.splitbaselineconfigs[4]) for i in range(self.config.atlas_folds[4])])

    def load_models(self,
            aal_paths = [],
            hammers_paths = [],
            rbn_paths = [],
            aicha_paths = [],
            harvard_paths = []
            ):
        # load aal
        for path,net in zip(aal_paths,self.aal_nets):
            net.load_state_dict(torch.load(path, map_location='cpu'))
        # load hammers
        for path,net in zip(hammers_paths,self.hammers_nets):
            net.load_state_dict(torch.load(path, map_location='cpu'))
        # load rbn
        for path,net in zip(rbn_paths,self.rbn_nets):
            net.load_state_dict(torch.load(path, map_location='cpu'))
        # load aicha
        for path,net in zip(aicha_paths,self.aicha_nets):
            net.load_state_dict(torch.load(path, map_location='cpu'))
        # load harvard
        for path,net in zip(harvard_paths,self.harvard_nets):
            net.load_state_dict(torch.load(path, map_location='cpu'))

    def forward(self,x):
        # x : [batchsize,feature_len]
        xs = []
        # aal prediction
        for net in self.aal_nets:
            xs.append(net(x).squeeze())
        # hammers prediction
        for net in self.hammers_nets:
            xs.append(net(x).squeeze())
        # rbn prediction
        for net in self.rbn_nets:
            xs.append(net(x).squeeze())
        # aicha prediction
        for net in self.aicha_nets:
            xs.append(net(x).squeeze())
        # harvard prediction
        for net in self.harvard_nets:
            xs.append(net(x).squeeze())
        x = torch.stack(xs)
        print(x)
        x = torch.mean(x,dim=0)
        return x

def merge_models():
    hammers_folds = [2,3,4,5,7,8,10]
    best_hammers_epoches = [797,921,914,780,796,781,922]      # Hammers Best
    hammers_paths = [
        "../model/Hammers/baseline_Hammers_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(hammers_folds,best_hammers_epoches)
    ]
    print(hammers_paths)
    rbn_folds = [3]
    best_rbn_epoches = [704]      # RBN Best
    rbn_paths = [
        "../model/rBN/baseline_rBN_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(rbn_folds,best_rbn_epoches)
    ]
    print(rbn_paths)
    aal_folds = [3]
    best_aal_epoches = [494]      # RBN Best
    aal_paths = [
        "../model/AAL/baseline_AAL_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(aal_folds,best_aal_epoches)
    ]
    print(aal_paths)
    aicha_folds = [3]
    best_aicha_epoches = [428]      # RBN Best
    aicha_paths = [
        "../model/AICHA_BIG/baseline_AICHA_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(aicha_folds,best_aicha_epoches)
    ]
    print(aicha_paths)
    harvard_folds = [3]
    best_harvard_epoches = [768]      # RBN Best
    harvard_paths = [
        "../model/HarvardOxford/baseline_HarvardOxford_lr1.e-3_dp0.6_fold%d_epoch%d.pth"%(x,y) for x,y in zip(rbn_folds,best_rbn_epoches)
    ]
    print(rbn_paths)
    config = MergeAtlasConfig(atlas_folds=[1,7,1,0,0])
    config.initialize()
    model = MergeAtlasBaseline(config)
    model.load_models(
            aal_paths = aal_paths,
            hammers_paths=hammers_paths,
            rbn_paths = rbn_paths,
            aicha_paths = [],
            harvard_paths = []
            )
    torch.save(model.state_dict(), "model.pth")
    # do a simple test here

def test():
    config = MergeAtlasConfig(atlas_folds=[1,7,1,0,0])
    config.initialize()
    print(config.__dict__)
    model = MergeAtlasBaseline(config)
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
    merge_models()
    test()
