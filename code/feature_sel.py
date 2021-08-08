from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from preprocess import load_data
import argparse
import numpy as np
import pandas as pd

# Read Data
# 加载数据
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',default="../data/train")
parser.add_argument('--input_info',default="../data/train_open.csv")
args = parser.parse_args()
print(args)
all_x, all_y = load_data(args.input_dir,
                    args.input_info
                    )
all_x = np.nan_to_num(all_x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
# mean = np.mean(all_x, axis=0)
# std = np.std(all_x, axis=0)
# all_x = (all_x - mean) / std
# all_x = np.nan_to_num(all_x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)

# define feature selection
fs,ps = f_classif(all_x, all_y)

info =  { "dim":range(len(fs)),"f":fs,"p":ps}
info_df = pd.DataFrame(info)
info_df = info_df.sort_values(by="f",ascending=False)
info_df.to_csv("../data/anova_info.csv",index=False)