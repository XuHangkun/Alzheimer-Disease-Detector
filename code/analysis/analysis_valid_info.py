import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch

def cal_precision(pred,true_label):
    """
    Args:
        pred : tensor, shape [N]
        true_label : tensor true label shape [N]
    return :
        precision
    """
    # assume pred.shape == true_label.shape
    pred = (pred >= 0.5)
    res = (pred == true_label)
    count = np.mean(res)
    return count

def cal_recall(pred,true_label):
    """
    Args:
        pred : tensor, shape [N]
        true_label : tensor true label shape [N]
    return :
        recall
    """
    pred = (pred > 0.5)
    recall = np.sum(pred)/np.sum(true_label)
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default="../model/valid_info.csv")
    parser.add_argument('--epoch',type=int,default=200)
    args = parser.parse_args()
    print(args)

    valid_info = pd.read_csv(args.input)

    info = {
        "label0_precision":[],"label1_precision":[],"label2_precision":[],
        "label0_recall":[],"label1_recall":[],"label2_recall":[],
        "label0_f1":[],"label1_f1":[],"label2_f1":[],"macro_f1":[],
        "mean_f1":[]
    }
    for epoch in range(1,args.epoch+1):
        if "epoch%d_true_label"%(epoch) not in valid_info.columns:
            continue
        true_label = valid_info["epoch%d_true_label"%(epoch)].to_numpy()
        for i in range(3):
            label_scores = valid_info["epoch%d_label%d_scores"%(epoch,i)].to_numpy()
            label_precision = cal_precision(label_scores,true_label==i)
            label_recall = cal_recall(label_scores,true_label==i)
            f1 = 2*label_precision*label_recall/(label_precision+label_recall)
            info["label%d_precision"%(i)].append(label_precision)
            info["label%d_recall"%(i)].append(label_recall)
            info["label%d_f1"%(i)].append(f1)
        mean_p = np.mean([info["label%d_precision"%(i)][-1] for i in range(2)])
        mean_r = np.mean([info["label%d_recall"%(i)][-1] for i in range(2)])
        mean_f1 = np.mean([info["label%d_f1"%(i)][-1] for i in range(2)])
        info["macro_f1"].append(2 * mean_p * mean_r / (mean_p + mean_r))
        info["mean_f1"].append(mean_f1)
    f1_fig = plt.figure()
    for i in range(3):
        plt.plot(info["label%d_f1"%(i)],label="Label %d"%(i))
    plt.plot(info["macro_f1"],label="Macro")
    plt.plot(info["mean_f1"],label="Mean")
    plt.xlabel("epoch",fontsize=14)
    plt.ylabel("F1",fontsize=14)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
