import os
import datetime
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import argparse
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from model import create_model
from preprocess import load_data, create_dataset , split_data, enhance_data
from utils import RAdam,Lookahead,eval_model,LabelSmoothingLoss

from sklearn.decomposition import PCA



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,default=7)
    # parameter of train
    parser.add_argument('--lr', type=float,default=1.e-4)
    parser.add_argument('--weight_decay', type=float,default=0)
    parser.add_argument('--smoothing', type=float, default = 0.05)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--epoch', type=int,default=100)
    parser.add_argument('--scheduler', action="store_true")
    parser.add_argument('--train_info_path',default="../model/train_info.csv")
    parser.add_argument('--valid_info_path',default="../model/valid_info.csv")
    # parameter of dataset
    #parser.add_argument('--atlas', default='all',choices=["all","AAL","AICHA","Brodmann","Gordon","Hammers",
    #                    "HarvardOxford","juelich","rBN","MIST","Schaefer","Yeo","Tian","Cerebellum"])
    parser.add_argument('--kth_fold',type=int,default=1)
    parser.add_argument('--atlas',nargs="+",default='AAL')
    parser.add_argument('--ctype', default='all',choices=["all","GMV","CT"])
    parser.add_argument('--input_dir',default="../data/train")
    parser.add_argument('--atlas_roi_path',default="../data/atlas_roi.csv")
    parser.add_argument('--input_info',default="../data/train_open.csv")
    parser.add_argument('--train_ratio',default=0.9,type=float)
    parser.add_argument('--enhance_n_aug',default=0,type=int)
    parser.add_argument('--enhance_tuning',default=0.015,type=float)
    # parameter of model
    parser.add_argument('--model_name',default="baseline",
            choices=["baseline","splitbaseline","chi2baseline","fbaseline","pcabaseline","transformer"])
    parser.add_argument('--dropout',default=0.1,type=float)
    parser.add_argument('--n_layers',default=6,type=int)
    parser.add_argument('--chi2_cut',default=0.5,type=float)
    parser.add_argument('--f_cut',default=150,type=float)
    parser.add_argument('--model_path',default="../model/model_epoch%d.pth")
    parser.add_argument('--save_per_epoch',default=1,type=int)
    parser.add_argument('--save_start_epoch',default=100,type=int)
    args = parser.parse_args()
    print(args)

    # Set Seed
    if args.seed is not None:
        print("Set random seed ")
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ????????????
    all_x, all_y, subjects = load_data(args.input_dir,
                        args.input_info
                        )

    # ???????????????
    all_x = np.nan_to_num(all_x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
    mean = np.mean(all_x, axis=0)
    std = np.std(all_x, axis=0)
    all_x = (all_x - mean) / std
    all_x = np.nan_to_num(all_x,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
    np.save('./mean.npy', mean)
    np.save('./std.npy', std)
    # define a valid item here
    valid_item = np.load("../data/train/Subject_0001.npy")
    valid_item = np.nan_to_num(valid_item,  nan=1.e-10, posinf=1.e-10, neginf=1.e-10)
    valid_item = (valid_item - mean) / std
    valid_item = torch.tensor([valid_item]).float().to(device)

    # ???????????????
    train_x,train_y,valid_x,valid_y,train_ids,valid_ids = split_data(all_x,all_y,args.train_ratio,args.kth_fold)
    valid_subjects = [subjects[item] for item in valid_ids]
    train_dataset = create_dataset(args.model_name ,train_x, train_y,n_aug=args.enhance_n_aug,tuning=args.enhance_tuning)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = create_dataset(args.model_name ,valid_x, valid_y,n_aug = args.enhance_n_aug,tuning=args.enhance_tuning/10)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=int(args.batch_size*len(valid_dataset)/len(train_dataset)),
        shuffle=True
        )

    print('Data has been loaded successfully!')

    model = create_model(
                    args.model_name,
                    in_dim = 50,
                    atlas_roi_path = args.atlas_roi_path,
                    atlas = args.atlas,
                    ctype = args.ctype,
                    dropout = args.dropout,
                    chi2_cut = args.chi2_cut,
                    f_cut = args.f_cut,
                    n_layers = args.n_layers,
                    device = device
                    )
    print('Create model successfully!')
    if os.path.exists(args.model_path):
        # ?????????????????????????????????
        model.load_state_dict(torch.load(args.model_path, map_location = 'cpu'))
        print('Load pretrained model from %'%(args.model_path))
    else:
        #for name, params in model.named_parameters():
        #    if 'bias' in name:
        #        pass
        #    else:
        #        torch.nn.init.kaiming_normal_(params,nonlinearity='relu')
        #print('Init model with kaiming normal')
        pass
    model.to(device)

    cross_entropy = nn.CrossEntropyLoss().to(device)
    smooth_cross_entropy = LabelSmoothingLoss().to(device)
    weight_p , bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = optim.SGD(
            [{'params':weight_p,'weight_decay':args.weight_decay},
                {'params':bias_p,'weight_decay':0}],
            lr=args.lr,momentum=0.9
            )
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=args.lr/10,last_epoch=-1)

    step = 0
    train_info = {"epoch":[] ,"train_loss":[], "valid_loss":[],"valid_item_score":[]}
    valid_info = {}
    for epoch in range(args.epoch):
        with tqdm(zip(train_data_loader,valid_data_loader), desc=f'Epoch {epoch + 1}') as epoch_loader:
            for train_data,valid_data in zip(train_data_loader,valid_data_loader):

                # ????????????
                model.train()
                inputs, labels = train_data
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                train_out_size = 1
                if len(outputs.shape) == 2:
                    train_loss = cross_entropy(outputs, labels.to(device))
                else:
                    train_loss = 0
                    train_out_size = outputs.size(0)
                    for i in range(outputs.size(0)):
                        train_loss += cross_entropy(outputs[i],labels.to(device))
                train_loss.backward()
                optimizer.step()
                if args.scheduler:
                    scheduler.step()

                # ????????????
                model.eval()
                inputs, labels = valid_data
                outputs = model(inputs.to(device))
                if len(outputs.shape) == 2:
                    valid_loss = cross_entropy(outputs, labels.to(device))
                else:
                    vaild_loss = 0
                    for i in range(outputs.size(0)):
                        valid_loss += cross_entropy(outputs[i],labels.to(device))
                # valid item
                valid_item_score = model(valid_item).squeeze()[0]

                epoch_loader.set_postfix(train_loss=f'{train_loss.item():.4f}',valid_loss=f'{valid_loss.item():.4f}')
                step += 1
                train_info["epoch"].append(step*1.0/(len(train_dataset)//args.batch_size))
                train_info["train_loss"].append(train_loss.item()/train_out_size)
                train_info["valid_loss"].append(valid_loss.item())
                train_info["valid_item_score"].append(valid_item_score.item())

        # save train info
        train_info_df = pd.DataFrame(train_info)
        train_info_df.to_csv(args.train_info_path,index=None)
        print('Train info saved to %s'%(args.train_info_path))
        # evaluate model on the valid data
        label_0_scores,label_1_scores,label_2_scores,true_label = eval_model(model,valid_data_loader)
        valid_info["epoch%d_label0_scores"%(epoch+1)] = label_0_scores
        valid_info["epoch%d_label1_scores"%(epoch+1)] = label_1_scores
        valid_info["epoch%d_label2_scores"%(epoch+1)] = label_2_scores
        valid_info["epoch%d_true_label"%(epoch+1)] = true_label
        valid_info_df = pd.DataFrame(valid_info)
        valid_info_df.to_csv(args.valid_info_path,index=None)
        if (epoch + 1)%args.save_per_epoch == 0 and (epoch + 1) > args.save_start_epoch:
           torch.save(model.state_dict(), args.model_path%(epoch))
           print('Save model to %s'%(args.model_path%(epoch)))
    print('End training')

if __name__ == "__main__":
    main()
