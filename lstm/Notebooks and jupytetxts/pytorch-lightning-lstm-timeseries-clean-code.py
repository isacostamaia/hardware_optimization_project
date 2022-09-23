# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="InjXlZH1iIEH"
# # Import dependencies

# + id="BM37JXb_gyfB" outputId="c2969a70-b749-4c37-9abc-c43482ce9bf4"
# Re-loads all imports every time the cell is ran. 
# %load_ext autoreload
# %autoreload 2

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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Plotting
# %matplotlib inline
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df, hosts_freqseries
from data_source.dataframe import hosts_timeseries

# -

# ## Retrieve Data

# +
# Set interval & filters
query_params = {
    # 'day', 'hour', 'minute'
    'interval': 'hour',
    # datetime
    'start_date': datetime.now() - timedelta(weeks =9),
    # datetime
    'end_date': None,
    # 'windows', 'linux'
    'os': 'windows',
    # List of host names
    'machines_to_include':None,
    # List of host names
    'machines_to_exclude': ['wip132dsy'],
    # Max number of records to fetch
    'limit': None
}

query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
records = query.all()

df_wks = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])


# -

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


# + [markdown] id="-kbn5QPQgyfK"
# # Prediction task
#
# We are going to predict hourly levels of global active power one step ahead.

# + [markdown] id="pu5mR7HfjKEJ"
# # TimeseriesDataset
#

# + id="DiXnMThtgyfL"
class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
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


# + [markdown] id="UxVFN7kaiNqR"
# # DataModule

# + id="XM6DMN6-iM2-"
class PowerConsumptionDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, seq_len = 1, batch_size = 128, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        '''
        
        #interpolate values and get same date intervals for all
        dfs, self.hostnames =  hosts_timeseries(df_wks)
        dfs = [a.rename(h) for a,h in zip(dfs,self.hostnames)]
        self.dataset = pd.concat(dfs,axis=1)

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
                                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = MyDataset(self.valid_all.float(), q = self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                drop_last=True,
                                num_workers = self.num_workers)

        return val_loader


# +
##debug
# 
# dm.val_dataloader().__len__()

# + [markdown] id="FuTfMKIajx_C"
# # Model
# Implement LSTM regressor using pytorch-lighting module

# + id="-SkFq9n_gyfr"
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
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 312)
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
        loss = self.criterion(y_hat, y)
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
        loss = self.criterion(y_hat, y)
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
        


# + [markdown] id="nbrpOYfc--uY"
# # Parameters

# + id="yQukeupz_Blu"
'''
All parameters are aggregated in one place.
This is useful for reporting experiment params to experiment tracking software
'''

p = dict(
    seq_len = 24,
    batch_size = 1, 
    criterion = nn.MSELoss(),
    max_epochs = 50,
    n_features = 312,
    hidden_size = 200,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)

# + [markdown] id="5YvfIXTFkXTa"
# # Train loop

# + id="evSq842Ygyfz" outputId="2cab8b21-a772-4d36-e082-565a4ef57463"
# %%time 

seed_everything(1)

csv_logger = CSVLogger('./', name='lstm', version='1'),


early_stop_callback = EarlyStopping(
   monitor='validation_loss_batch',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='max'
)


trainer = Trainer(
    max_epochs=p['max_epochs'],
    logger=csv_logger,
    gpus=1,
#     row_log_interval=1,
    progress_bar_refresh_rate=2,
#     callbacks=[early_stop_callback],
)

model = LSTMRegressor(
    n_features = p['n_features'],
    hidden_size = p['hidden_size'],
    seq_len = p['seq_len'],
    batch_size = p['batch_size'],
    criterion = p['criterion'],
    num_layers = p['num_layers'],
    dropout = p['dropout'],
    learning_rate = p['learning_rate']
)

dm = PowerConsumptionDataModule(
    seq_len = p['seq_len'],
    batch_size = p['batch_size']
)

trainer.fit(model, dm)
# trainer.test(model, datamodule=dm)

# +
# # It is implemented the built-in function for finding the
# # optimal learning rate. Source: https://arxiv.org/pdf/1506.01186.pdf
# lr_finder = trainer.tuner.lr_find(model, 
#                         min_lr=0.0005, 
#                         max_lr=0.005,
#                         mode='linear')

# # Plots the optimal learning rate
# fig = lr_finder.plot(suggest=True)
# fig.show()

# +
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def get_scaled_predictions():
    train_window = p['seq_len']
    fut_pred = len(dm.test_all)

    test_inputs = (dm.train_all[-train_window:])
    
    if device == 'cuda':
        test_inputs = test_inputs.cuda().to(device)

    model.eval()

    #  Get predictions on test data
    for i in range(fut_pred):
        #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
        seq = (test_inputs[-train_window:]).unsqueeze(0).float() 
        
        if device == 'cuda':
            seq = seq.cuda().to(device)
        
        with torch.no_grad():
            test_preds = model(seq)
            #if it is three dimensional, take out the first dimension (that will be equal 1, the batch_size)
            test_preds = test_preds.squeeze()
            #get last prediction and concatenate with pred tensor
            test_inputs = torch.cat((test_inputs,test_preds[-1].unsqueeze(0)),0)



    tests_fin = test_inputs[-len(dm.test_all):]
    return tests_fin

def treat_predictions(tests_fin):
    
    #transform predictions tensor to dataframe
    tests_fin.shape # len(Test_all), num_machines
    preds_df = pd.DataFrame(tests_fin.cpu().numpy())
    preds_df_unscaled = preds_df.copy()
    
    #undo predictions rescaling
    preds_array_unscaled = dm.scaler.inverse_transform(preds_df)
    dic = {i:h for i,h in zip(np.arange(0,preds_df.columns.stop), dm.hostnames)}
    preds_df = preds_df.rename(columns=dic)
    
    #replace predictions unscaled values in dataframe
    preds_df= pd.DataFrame(preds_array_unscaled, columns= dm.hostnames)
    
    return preds_df

df_preds = treat_predictions(get_scaled_predictions())


# -

def plot(hostnames,dataset,preds_df,len_train):
    fig,axs = plt.subplots(len(hostnames),1,figsize=(20,30),sharex=True, sharey=True)

    for ax,h in zip(axs, hostnames):
        
        #ground truth
        df_mach = dataset.loc[:, h].copy()
        df_mach = df_mach.iloc[:len_train+len(preds_df)]
        gd_th_interp = df_mach.values
        x_gd_th = np.arange(0,len_train+len(preds_df),1)
        
        #predictions
        pred = preds_df.loc[:,h]
        x_pred = np.arange(len_train, len_train+len(pred),1)
        
        #plot
        ax.plot(x_gd_th,gd_th_interp,  label='Ground Truth', )
        ax.plot(x_pred, pred, 'r--', label='Test Predictions')
        ax.set_title(h)
        
        ax.set_ylim([-10,75])
        
    fig.subplots_adjust(hspace = 0.2)


plot(dm.hostnames[:10],dm.dataset,df_preds,len(dm.train_all))

plot(dm.hostnames[30:40],dm.dataset,df_preds,len(dm.train_all))

# +
#compute distance between predictions and groud truth

gd_truth_test = (dm.dataset.iloc[len(dm.train_all):len(dm.train_all)+len(dm.test_all), :])
gd_truth_test = gd_truth_test.to_numpy()

preds_test = df_preds.to_numpy()

distance = dtw(gd_truth_test, preds_test)

#[compute distance after and save results in lstm/0 with predictions images]

# +
#[do train and valid loss plots]

# + [markdown] id="elE3vzV-yoI5"
#
# # Plot report

# + id="QKAKWR3IlTk9" outputId="b06ed628-6564-482a-970b-319caa25cb1b"
metrics = pd.read_csv('./lstm/1/metrics.csv')
train_loss = metrics[['Train_loss_epoch', 'step', 'epoch']][~np.isnan(metrics['Train_loss_epoch'])]
val_loss = metrics[['Validation_loss_epoch', 'epoch']][~np.isnan(metrics['Validation_loss_epoch'])]
# test_loss = metrics['test_loss'].iloc[-1]

# fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
# axes[0].set_title('Train loss per epoch')
# axes[0].plot(train_loss['epoch'], train_loss['Train_loss_epoch'])
# axes[1].set_title('Validation loss per epoch')
# axes[1].plot(val_loss['epoch'], val_loss['Validation_loss_epoch'], color='orange')
# plt.show(block = True)


fig, axes = plt.subplots(1, 1, figsize=(16, 5), dpi=100)
axes.set_title('Loss per epoch')
axes.plot(train_loss['epoch'], train_loss['Train_loss_epoch'], label = "Train")
axes.plot(val_loss['epoch'], val_loss['Validation_loss_epoch'], color='orange', label = "Validation")

plt.legend()
plt.show(block = True)

# -

train_loss
# val_loss


