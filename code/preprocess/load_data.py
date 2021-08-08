import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(subject_dir, csv_path):
    """load data

    Args:
        subject_dir : path for data
        csv_path : path for train info
        atlas_roi_csv : path for atlas info
        atlas : you can choose [all,AAL,AICHA,Brodmann,Gordon,Hammers,
                        HarvardOxford,juelich,rBN,MIST,Schaefer,Yeo,Tian,Cerebellum]
        type : you can choose [all,GMV,CT]
    """
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)
    x = []
    y = []
    print("Load data Here")
    for subject in tqdm(subjects):
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[subject.split('.')[0]]
            label = int(row['Label'])
            tiv = float(row['TIV'])
            xx = np.load(features_path)
            # xx[:13925] = xx[:13925]/tiv
            x.append(xx)
            y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y, subjects

def test():
    atlas_roi_df = pd.read_csv("../data/atlas_roi.csv")
    print(cal_x_ids(atlas_roi_df,atlas="all",ctype="all"))
    print(cal_x_ids(atlas_roi_df,atlas="AAL",ctype="all"))
    print(cal_x_ids(atlas_roi_df,atlas="AAL",ctype="GMV"))
    print(cal_x_ids(atlas_roi_df,atlas="AAL",ctype="CT"))

if __name__ == "__main__":
    test()
