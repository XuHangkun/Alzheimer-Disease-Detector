import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from model import create_model
from preprocess import load_data, create_dataset , split_data
from utils import RAdam,Lookahead


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,default=7)
    # parameter of train
    parser.add_argument('--lr', type=float,default=1.e-2)
    parser.add_argument('--weight_decay', type=float,default=1.e-2)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--epoch', type=int,default=10)
    parser.add_argument('--train_info_path',default="../model/train_info.csv")
    # parameter of dataset
    parser.add_argument('--input_dir',default="../data/train")
    parser.add_argument('--input_info',default="../data/train_open.csv")
    parser.add_argument('--train_ratio',default=0.8,type=float)
    # parameter of model
    parser.add_argument('--model_name',default="baseline",choices=["baseline"])
    parser.add_argument('--model_path',default="../model/baseline_model.pkl")
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # 加载数据
    all_x, all_y = load_data(args.input_dir, args.input_info)

    # 数据预处理
    all_x = np.nan_to_num(all_x, nan=0, posinf=0, neginf=0)
    mean = np.mean(all_x, axis=0)
    std = np.std(all_x, axis=0)
    all_x = (all_x - mean) / std
    all_x = np.nan_to_num(all_x, nan=0, posinf=0, neginf=0)

    # 生成数据集
    train_x,train_y,valid_x,valid_y = split_data(all_x,all_y,args.train_ratio)
    train_dataset = create_dataset(args.model_name ,train_x, train_y)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = create_dataset(args.model_name ,valid_x, valid_y)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, 
        batch_size=int(args.batch_size*len(valid_dataset)/len(train_dataset)), 
        shuffle=True
        )

    print('Data has been loaded successfully!')

    model = create_model(args.model_name)
    print('Create model successfully!')
    if os.path.exists(args.model_path):
        # 如果模型存在，则加载之
        model.load_state_dict(torch.load(args.model_path, map_location =device))
        print('Load pretrained model from %'%(args.model_path))
    else:
        pass
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # 训练次数设为10次
    step = 0
    train_info = {"step":[] ,"train_loss":[], "valid_loss":[]}
    for epoch in range(args.epoch):
        with tqdm(zip(train_data_loader,valid_data_loader), desc=f'Epoch {epoch + 1}') as epoch_loader:
            for train_data,valid_data in zip(train_data_loader,valid_data_loader):

                # 训练模型
                model.train()
                inputs, labels = train_data
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                train_loss = criterion(outputs, labels.to(device))
                train_loss.backward()
                optimizer.step()

                # 验证模型
                model.eval()
                inputs, labels = valid_data
                outputs = model(inputs.to(device))
                valid_loss = criterion(outputs, labels.to(device))

                epoch_loader.set_postfix(train_loss=f'{train_loss.item():.4f}',valid_loss=f'{valid_loss.item():.4f}')
                step += 1
                train_info["step"].append(step)
                train_info["train_loss"].append(train_loss.item())
                train_info["valid_loss"].append(valid_loss.item())
        # save train info
        train_info_df = pd.DataFrame(train_info)
        train_info_df.to_csv(args.train_info_path,index=None)
        print('Train info saved to %s'%(args.train_info_path))

    print('End training')
    np.save('./mean.npy', mean)
    np.save('./std.npy', std)
    # torch.save(model.state_dict(), args.model_path)
    print('Save model to %s'%(args.model_path))

if __name__ == "__main__":
    main()