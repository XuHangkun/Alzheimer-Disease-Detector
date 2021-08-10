#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from model_service.pytorch_model_service import PTServingBaseService
from create_model import create_model

class PredictService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        super(PredictService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(self.model_path))
        model = create_model(
                "mergeaalhammersrbn",
                atlas_roi_path = os.path.join(dir_path,"atlas_roi.csv"),
                atlas_folds = [1,7,1,0,0]
                )
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
