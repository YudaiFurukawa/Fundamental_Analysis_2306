# Import comet_ml at the top of your file
from comet_ml import Experiment
from datasets import Custom_dataset_220919

import pandas as pd
import numpy as np
import os, sys,random
from collections import deque
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats.stats import pearsonr
from model import Model_Attention_221029 as net
import re, csv
import itertools

import utils



### Torch
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory':True}
val_params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 8,
            #   'drop_last':True,
              'pin_memory':True}
LEARNING_RATE = 1e-3

def dataset(train_date_till,test_date_from):
    folder = './data/transformed_data/'
    df_x = pd.read_csv(folder+'data_x_transformed.csv')
    df_target = pd.read_csv(folder+'data_y_transformed.csv')
    
    ### For Testing
    # df_x = df_x[df_x['symbol']=='BAC']
    # df_target = df_target[df_target['symbol'].isin(['BAC'])]
    ###

    print("df_x[['date','symbol']]",df_x[['date','symbol']])
    print("df_target[['date','symbol']]",df_target[['date','symbol']])

    df_x_train = df_x[df_x['date']<train_date_till].copy()
    df_target_train = df_target[df_target['date']<train_date_till].copy()
    df_x_test = df_x[df_x['date']>=test_date_from].copy()
    df_target_test = df_target[df_target['date']>=test_date_from].copy()
    dataset_train = Custom_dataset_220919(df_x=df_x_train,df_y=df_target_train,window=12,target_col='TR_750d_rank')
    dataset_test = Custom_dataset_220919(df_x=df_x_test,df_y=df_target_test,window=12,target_col='TR_750d_rank')
    return dataset_train,dataset_test
def dataloader(dataset_train, dataset_test, params, val_params):
    train_loader = data.DataLoader(dataset_train, **params)
    val_loader = data.DataLoader(dataset_test, **val_params)
    return train_loader,val_loader
def train_one_epoch(model,train_loader,device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y = y.unsqueeze(1)
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i,"loss:", loss.item())
    return model,loss
def evaluate_one_epoch(model,val_loader,device):
    model.eval()

    for i, (x, y) in enumerate(val_loader):
        if i < 1:
            y = y.unsqueeze(1)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            print("evaluate: ", i,loss.item())
    return model,loss
def main(folder_saved_model = './saved_model/'):
    '''
    Training Loop 
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    '''
    # Create an experiment with your api key
    experiment = Experiment(
            api_key="3dZ6osa1d98oDZoB5U0ieOfqy",
            project_name="compai",
            workspace="yudaifurukawa",
        )
    experiment.log_parameters(params)
    experiment.log_parameters(val_params)

    ### Prepare Data Loaders
    dataset_train, dataset_test = dataset(train_date_till='2018-01-01',test_date_from='2018-01-01')
    datalaoder_train, dataloader_test = dataloader(dataset_train, dataset_test,1 params, val_params)
    print(f"train: {len(dataset_train)} test: {len(dataset_test)}")

    ### Train Model
    best_loss_val = np.inf
    model = net(in_shape=dataset_train[0][0].shape,out_shape=1).to(device)

    try:
        model.load_state_dict(torch.load(folder_saved_model+'model.pth'))
        print("model loaded")
    except:
        print("model not loaded")
    
    for epoch in range(10000):
        print('Epoch: ',epoch)
        with experiment.train():
            _,loss = train_one_epoch(model=model,train_loader=datalaoder_train,device=device)
            experiment.log_metric("loss", loss.item(), epoch=epoch)
        with experiment.validate():
            _,loss_val = evaluate_one_epoch(model=model,val_loader=dataloader_test,device=device)
            experiment.log_metric("loss", loss_val.item(), epoch=epoch)
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(model.state_dict(), folder_saved_model+'model.pth')
            print('model saved')

if __name__ == '__main__':
    folder_saved_model = './saved_model/221129/'
    main(folder_saved_model=folder_saved_model)
    # dataset(train_date_till='2018-01-01',test_date_from='2018-01-01')
# dataset_train, dataset_test = dataset(train_date_till='2020-01-01',test_date_from='2020-01-01')
# print(len(dataset_train),len(dataset_test))
# print(dataset_test[3])
