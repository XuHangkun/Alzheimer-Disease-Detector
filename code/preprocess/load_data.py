import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(subject_dir, csv_path):
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
            x.append(np.load(features_path))
            y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y