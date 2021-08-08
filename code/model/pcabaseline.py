import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .useless_ids import useless_ids
from collections import Iterable

class PCABaselineConfig:
    def __init__(self,
            in_dim = 25,
            out_dim = 3,
            dropout = 0.1
        ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
    def initialize(self):
        self.n_hidden = min(self.in_dim//2 , 512)

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

class PCABaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.in_dim, self.config.n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(self.config.dropout),
            FullResLayer(self.config.n_hidden,self.config.dropout),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_hidden, self.config.out_dim),
            nn.Softmax(dim=-1)
        )

    def _preprocess(self,x):
        pass

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
        x = self.net(x)
        return x
