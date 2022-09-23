
from time import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format

from IPython.display import display

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_forecasting.metrics import MAPE
from soft_dtw_cuda import SoftDTW

from .retrieve_data import get_df

# ## Split Data function

def split_data(tensor):
    '''
        returns train, test dataframes.
        Obs:  (index where test starts is clearly len(train)+1)
    '''
    per = 0.7 #percentage train
    len_train = int(per*len(tensor))
    train = tensor[:len_train,:]
    len_test = int((len(tensor) - len_train)/2)
    test = tensor[len_train:len_train+len_test, :]
    valid = tensor[len_train+len_test: , :]
    return train, test, valid



class MyDataset(Dataset):
    def __init__(self, data, q):
        self.data = torch.Tensor(data)
        self.window = q

    def __len__(self):
        return len(self.data) -  self.window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+1: index + self.window + 1]
        return x , y


class DataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, seq_len = 1, batch_size = 128, num_workers=0, pin_memory = False):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dfs, self.hostnames =  get_df()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        '''
        
        #interpolate values and get same date intervals for all

        self.dfs = [a.rename(h) for a,h in zip(self.dfs,self.hostnames)]
        self.dataset = pd.concat(self.dfs,axis=1)

        #scale values
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        dataset_array_scaled = self.scaler.fit_transform(self.dataset)
        dataset_scaled = self.dataset.copy()
        dataset_scaled.loc[:,:] = dataset_array_scaled


        tensor_dataset = torch.tensor(dataset_scaled.values)

        self.train_all, self.test_all, self.valid_all = split_data(tensor_dataset)


    def train_dataloader(self):
        train_dataset = MyDataset(self.train_all.float(), q = self.seq_len)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size = self.batch_size, 
                                                   shuffle = False,
                                                   drop_last=True,
                                                  num_workers = self.num_workers, 
                                                  pin_memory = self.pin_memory)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = MyDataset(self.valid_all.float(), q = self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                drop_last=True,
                                num_workers = self.num_workers,
                                pin_memory = self.pin_memory)

        return val_loader




class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion,
                 epochs,
                 start_date, 
                 end_date, 
                 num_weeks):

        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.start_date = start_date
        self.num_weeks_train = num_weeks
        self.end_date = end_date

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, self.n_features)
        self.save_hyperparameters()
        
        self.previous_hidden = None 


    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(x,self.previous_hidden)
        y_pred = self.linear(lstm_out) #lstm_out[:,-1]
        self.previous_hidden = tuple([h.data for h in hidden])
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)


        #MAPE loss
        if isinstance(self.criterion, MAPE):
            loss = torch.zeros(1)
            # loss = loss.cuda().to(self.device)
            loss = loss.type_as(x) 
            for b in range(self.batch_size):
                out_one_of_batch = y_hat[b,:,:]
                y_one_of_batch = y[b,:,:]
                loss  = loss + self.criterion(out_one_of_batch, y_one_of_batch)
            loss = loss/self.batch_size

        else:
            loss = self.criterion(y_hat, y)
            if isinstance(self.criterion, SoftDTW):
                loss = torch.mean(loss)

        self.log('train_loss_batch', loss)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like: 
        # [{'loss':x}, {'loss':x}, ...]

        # Option 1
        # We can unfold the loss, then just take the mean
        loss_epoch = []
        for out in outputs:
            loss_epoch.append(out['loss'])
        loss_epoch = torch.mean(torch.stack(loss_epoch), dim=0)
        print(f"Train Loss: {loss_epoch}")

        # Save the metric
        self.log('Train_loss_epoch', loss_epoch, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        #MAPE loss
        if isinstance(self.criterion, MAPE):
            loss = torch.zeros(1)
            loss = loss.type_as(x)  
            for b in range(self.batch_size):
                out_one_of_batch = y_hat[b,:,:]
                y_one_of_batch = y[b,:,:]
                loss  = loss + self.criterion(out_one_of_batch, y_one_of_batch)
            loss = loss/self.batch_size

        else:
            loss = self.criterion(y_hat, y)
            if isinstance(self.criterion, SoftDTW):
                loss = torch.mean(loss)


        self.log('validation_loss_batch', loss)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        # This function recevies as parameters the output from "validation_step()"
        # Outputs is a list which contains a dictionary like: 
        # [{'loss':x}, {'loss':x}, ...]

        # Option 1
        # We can unfold the loss, then just take the mean
        loss_epoch = []
        for out in outputs:
#             print("loss_epoch shape",loss_epoch.shape)
            loss_epoch.append(out['loss'])
        loss_epoch = torch.mean(torch.stack(loss_epoch), dim=0)
        print(f"Validation Loss: {loss_epoch}")

        # Save the metric
        self.log('Validation_loss_epoch', loss_epoch, prog_bar=True)

