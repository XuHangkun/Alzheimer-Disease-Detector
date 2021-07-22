import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import torch.utils.data as data

class BaselineDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        xi = self.x[index]
        yi = self.y[index]
        return xi, yi

    def __len__(self):
        return len(self.y)