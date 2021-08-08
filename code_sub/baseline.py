import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class BaselineConfig:
    def __init__(self,in_dim = 28169,
        n_hidden_1 = 2048,
        n_hidden_2 = 256,
        out_dim = 3,
        dropout = 0.1
        ):
        self.in_dim = in_dim
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.out_dim = out_dim
        self.dropout = dropout

class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config.in_dim, self.config.n_hidden_1)
        self.layer2 = nn.Linear(self.config.n_hidden_1, self.config.n_hidden_2)
        self.layer3 = nn.Linear(self.config.n_hidden_2, self.config.out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.softmax = nn.Softmax(dim=-1)

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
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

def test():
    config = BaselineConfig()
    print(config.__dict__)
    model = Baseline(config)
    # model.load_state_dict(torch.load("model.pth", map_location ='cpu'))
    model.eval()
    x = np.load("../data/train/Subject_0001.npy")
    mean = np.load("mean.npy")
    std = np.load("std.npy")
    x = (x - mean)/std
    x = torch.tensor(x).unsqueeze(0).float()
    print(x.shape)
    y = model(x)
    print(y)

if __name__ == "__main__":
    test()
