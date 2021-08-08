import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

def TrainPCA(X,chi2_cut=0.1,n_comp=50):
    """Train PCA
    """
    chi2_info = pd.read_csv("../data/chi2_info.csv")
    sel_dims = chi2_info[chi2_info["chi2"] > chi2_cut]["dim"].to_numpy()
    new_x = X[:,sel_dims]
    pca = PCA(n_components=n_comp)
    pca.fit(new_x)
    red_x = pca.transform(new_x)
    return pca, red_x

def main():
    train_open_data = pd.read_csv("../data/train_open.csv")
    features = []
    for i in tqdm(range(len(train_open_data))):
        file_name = os.path.join("../data/train","%s.npy"%(train_open_data["new_subject_id"][i]))
        x = np.load(file_name)
        x = np.nan_to_num(x,  nan=0, posinf=0, neginf=0)
        label = train_open_data["Label"][i]
        features.append(x)
    features = np.array(features)
    pca,red_x = TrainPCA(features)
    print(pca.explained_variance_ratio_)
    print("Sum : %.4f"%(sum(pca.explained_variance_ratio_)))

if __name__ == "__main__":
    main()
