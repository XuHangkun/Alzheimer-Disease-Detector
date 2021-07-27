#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
# from model_service.pytorch_model_service import PTServingBaseService

class SplitBaselineConfig:
    def __init__(self,
            atlas_roi_path="atlas_roi.csv",
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
        self.x_ids = np.array([  708,   709,   710,   711,   712,   713,   714,   715,   716,
         717,   718,   719,   720,   721,   722,   723,   724,   725,
         726,   727,   728,   729,   730,   731,   732,   733,   734,
         735,   736,   737,   738,   739,   740,   741,   742,   743,
         744,   745,   746,   747,   748, 14711, 14712, 14713, 14714,
       14715, 14716, 14717, 14718, 14719, 14720, 14721, 14722, 14723,
       14724, 14725, 14726, 14727, 14728, 14729, 14730, 14731, 14732,
       14733, 14734, 14735, 14736, 14737, 14738, 14739, 14740, 14741,
       14742, 14743, 14744, 14745, 14746, 14747, 14748, 14749, 14750,
       14751])
        self.in_dim = len(self.x_ids)
        self.n_hidden_1 = self.in_dim // 2
        self.n_hidden_2 = self.in_dim // 2

    def cal_x_ids(self):
        atlas_roi_df = pd.read_csv(self.atlas_roi_path)
        atlas = self.atlas.lower()
        ctype = self.ctype.lower()
        if atlas == "all":
            if ctype == "gmv":
                return np.array([atlas_roi_df["dim"][i] for i in range(len(atlas_roi_df)) if "mesh" in atlas_roi_df["Atlas"][i]])
            elif ctype == "ct":
                return np.array([atlas_roi_df["dim"][i] for i in range(len(atlas_roi_df)) if "mesh" not in atlas_roi_df["Atlas"][i]])
            else:
                return atlas_roi_df["dim"].to_numpy()
        else:
            x_ids = []
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
            return np.array(x_ids)

class SplitBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_ids = config.x_ids
        self.layer1 = nn.Linear(self.config.in_dim, self.config.n_hidden_1)
        self.layer2 = nn.Linear(self.config.n_hidden_1, self.config.n_hidden_2)
        self.layer3 = nn.Linear(self.config.n_hidden_2, self.config.out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.softmax = nn.Softmax(dim=-1)

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
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

class PredictService:
    def __init__(self, model_name, model_path):
        # super(PredictService, self).__init__(model_name, model_path)
        self.model_path = model_path
        config = SplitBaselineConfig(atlas="Brodmann")
        config.initialize()
        print(config.__dict__)
        model = SplitBaseline(config)
        model.load_state_dict(torch.load(model_path, map_location ='cpu'))
        model.eval()
        self.model = model
        self.load_preprocess()

    def load_preprocess(self, mean_name='mean.npy', std_name='std.npy'):
      dir_path = os.path.dirname(os.path.realpath(self.model_path))

      mean_path = os.path.join(dir_path, mean_name)
      std_path = os.path.join(dir_path, std_name)
      self.mean = np.load(mean_path)
      self.std = np.load(std_path)

    def _preprocess(self, data):
        print('pre_data:{}'.format(data))
        preprocessed_data = {}

        for d in data:
            for k, v in data.items():
                for file_name, features_path in v.items():
                    x = np.load(features_path)
                    # deploy environment numpy version
                    x = (x - self.mean) / self.std
                    x = np.nan_to_num(x)
                    x[x>1000000] = 0
                    x[x<-1000000] = 0
                    x = torch.from_numpy(x).to(torch.float32)
                    preprocessed_data[k] = x
        return preprocessed_data

    def _postprocess(self, data):
        print('post_data:{}'.format(data))
        infer_output = {}

        for output_name, result in data.items():
            infer_output['scores'] = result.tolist()

        return infer_output

if __name__ == "__main__":
    PredictService("baseline","model.pth")
