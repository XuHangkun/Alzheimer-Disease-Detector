from sklearn import svm, metrics,datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn import tree

def TrainPCA(X,chi2_cut=0.75,n_comp=250):
    """Train PCA
    """
    chi2_info = pd.read_csv("../data/chi2_info.csv")
    sel_dims = chi2_info[chi2_info["chi2"] > chi2_cut]["dim"].to_numpy()
    new_x = X[:,sel_dims]
    pca = PCA(n_components=n_comp)
    pca.fit(new_x)
    red_x = pca.transform(new_x)
    return pca, new_x
def cal_x_ids(
    chi2_info_path="../data/chi2_info.csv",
    chi2_cut = 0.75
    ):
    chi2_info = pd.read_csv(chi2_info_path)
    chi2_info = chi2_info[chi2_info["chi2"] > chi2_cut]
    dims = []
    chi2s = []
    for i in range(len(chi2_info)):
        chi2 = "%.7f"%(chi2_info["chi2"][i])
        if chi2 not in chi2s:
            chi2s.append(chi2)
            dims.append(chi2_info["dim"][i])
    print("Num of Dims : %d"%(len(dims)))
    return np.array(dims)

# load data here
train_open_data = pd.read_csv("../data/train_open.csv")
idxs = cal_x_ids()
features = []
labels = []
for i in tqdm(range(len(train_open_data))):
    file_name = os.path.join("../data/train","%s.npy"%(train_open_data["new_subject_id"][i]))
    x = np.load(file_name)
    x = np.nan_to_num(x,  nan=0, posinf=0, neginf=0)
    label = train_open_data["Label"][i]
    features.append(x)
    labels.append(label)
features = np.array(features)
features = np.nan_to_num(features,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features = (features - mean) / std
features = np.nan_to_num(features,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
features = features[:,idxs]
labels = np.array(labels)

# PCA 
# pca,red_x = TrainPCA(features)
# print(pca.explained_variance_ratio_)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(predicted)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")