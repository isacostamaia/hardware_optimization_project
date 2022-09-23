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

# +
import torch
from torch import nn, optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import datetime, timedelta
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df, hosts_freqseries
from data_source.dataframe import hosts_timeseries


mpl.rcParams['figure.figsize'] = [12,8]
viz_dict = {
    'axes.titlesize':18,
    'axes.labelsize':16,
}
sns.set_context("notebook", rc=viz_dict)
sns.set_style("whitegrid")

# +
# %%time

list_mach = ['wip196dsy','ssdhwip017dsy','wip212dsy','wip214dsy','pool27dsy','pool36dsy']

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
# -

df_wks = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

df_one = df_wks[df_wks.hostname=='ssdhwip017dsy']

# +
#interpolate values and get same date intervals for all
dfs, hostnames =  hosts_timeseries(df_wks)
dfs = [a.rename(h) for a,h in zip(dfs,hostnames)]
dataset = pd.concat(dfs,axis=1)

print("Dataset before scaling: \n", dataset.head(15))
#scale values
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset_array_scaled = scaler.fit_transform(dataset)
dataset_scaled = dataset.copy()
dataset_scaled.loc[:,:] = dataset_array_scaled
print("Dataset after scaling: \n", dataset_scaled.head(15))

tensor_dataset = torch.tensor(dataset_scaled.values)


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

train_all, test_all, valid_all = split_data(tensor_dataset)

# -

dataset.info()


# +
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
    
def get_batches_dataloader(data, window, batch_size):
    dataset = MyDataset(data.float(), q = window)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last=True)


# +
class NeuralNet(pl.LightningModule):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 learning_rate, train_all, valid_all, test_all, sequence_len = 3, batch_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)
        
        self.batch_size = batch_size
        
        self.hs = None
        self.sequence_len = sequence_len
        
        #datasets
        self.train_all = train_all
        self.test_all = test_all,
        self.valid_all = valid_all
        
        #losses
        self.train_loss = []
        self.valid_loss = []
        
        #hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x, hs):
        print("\n FORWARD     ********** \n")
        print("X SHAPE", x.shape)
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
        print("out shape ",out.shape)
        out = out.squeeze() # out.shape = (seq_len, hidden_size)    
        print("out shape after reshape",out.shape)
        out = self.fc(out)
        
        return out, hs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        print("\n TRAINING STEP     ********** \n")
        
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        print("x shape training step: ", x.shape)
        print("y shape training step: ", y.shape)

        # Feed the model and catch the prediction
#         x = x.unsqueeze(0)

        out, self.hs = self.forward(x,self.hs) 
        self.hs = tuple([h.data for h in self.hs])
        print("out shape ", out.shape)
        
        # Calculates loss for the current batch
        criterion = nn.MSELoss()
        loss =  criterion(out.view(self.batch_size,self.sequence_len, y.shape[-1]), y)
        print("format out in loss: ", out.view(self.batch_size,self.sequence_len, y.shape[-1]).shape)
        print("format y in loss: ", y.shape)

        # Calculates accuracy for current batch
#         train_acc_batch = self.train_accuracy(y_pred, y)

        # Save metrics for current batch
#         self.log('train_acc_batch', train_acc_batch)
        self.log('train_loss_batch', loss)
        
        return {'loss' : loss, 'y_pred' : out, 'y_true' : y}
    
    def training_epoch_end(self, training_step_outputs):
        #lets hope pl.ligh does this at each epoch (by the name of function i guess this is the case)
        t_loss = 0
        for pred in training_step_outputs:
            t_loss += pred['loss'].item()
          
        #accumulated sum of the epoch
        self.train_loss.append(np.mean(t_loss))#mean of scalar is the scalar itself    

    def validation_step(self, batch, batch_idx):
        print("\n VALIDATION STEP     ********** \n")

        
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        print("x shape validation step", x.shape)    
        print("y shape validation step: ", y.shape)

        # Feed the model and catch the prediction (no need to set the model as "evaluation" mode)
#         x = x.unsqueeze(0)

        out, self.hs = self.forward(x,self.hs)
        print("out shape ", out.shape)

        self.hs = tuple([h.data for h in self.hs])
        
        # Calculate loss for the current batch
        criterion = nn.MSELoss()
        print("out shape validation step", out.shape)
        print("y shape validation step", y.shape)
        loss =  criterion(out.view(self.batch_size,self.sequence_len, y.shape[-1]), y)
        print("format out in loss: ", out.view(self.batch_size,self.sequence_len, y.shape[-1]).shape)
        print("format y in loss: ", y.shape)

        # Calculates accuracy for the current batch
#         val_acc_batch = self.val_accuracy(y_pred, y)
        
        # Save metrics for current batch
#         self.log('val_acc_batch', val_acc_batch, prog_bar=False)
        self.log('val_loss_batch', loss, prog_bar=False)

        return {'loss' : loss, 'y_pred' : out, 'target' : y}
    
    def validation_epoch_end(self, validation_step_outputs):
        #lets hope pl.ligh does this at each epoch (by the name of function i guess this is the case)
        v_loss = 0
        for pred in validation_step_outputs:
            v_loss += pred['loss'].item()
          
        #accumulated sum of the epoch
        self.valid_loss.append(np.mean(v_loss))#mean of scalar is the scalar itself     
    
    def test_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        
        # Feed the model and catch the prediction (no need to set the model as "evaluation" mode)
#         x = x.unsqueeze(0)

        out, self.hs = self.forward(x,self.hs)
        self.hs = tuple([h.data for h in self.hs])
        
        # Calculate loss for the current batch
        criterion = nn.MSELoss()
        loss =  criterion(out.view(self.batch_size,self.sequence_len, y.shape[-1]), y)

        # Calculates accuracy for the current batch
#         val_acc_batch = self.val_accuracy(y_pred, y)
        
        # Save metrics for current batch
#         self.log('val_acc_batch', val_acc_batch, prog_bar=False)
        self.log('test_loss_batch', loss, prog_bar=False)

        return {'loss' : loss, 'y_pred' : out, 'target' : y}
    

    
    
    
#     def prepare_data(self):
#         # Here you can do wherever you need to have your data ready
#         # In this case I'm just downloading the dataset
#         self.x, self.y = load_breast_cancer(return_X_y=True)

#     def setup(self, stage=None):
#         # Setup function is key when handling dataloaders inside the class since
#         # this function is triggered when training or testing the model by passing the
#         # right datasets (training, validation and testing)
        

#         # Assign train/val datasets for using in dataloaders
#         if stage == 'fit' or stage is None:
            
            
            
#         # Assign test dataset for using in dataloader(s)
#         if stage == 'test' or stage is None:
            
        

    def train_dataloader(self):
        print("TRAIN DATALOADER *******\n")

        # Called when training the model
        self.train_dataset = MyDataset(train_all.float(), sequence_len)
        for x, y in torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False):
            print("y shape train_dataloader: ", y.shape)
            break
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False)

    def val_dataloader(self):
        print("VAL DATALOADER *******\n")
        # Called when evaluating the model (for each "n" steps or "n" epochs)
        self.val_dataset = MyDataset(valid_all.float(), sequence_len)
        for x, y in torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False):
            print("y shape val_dataloader: ", y.shape)
            print("x shape val_dataloader: ", x.shape)
            print("batch_size = ", self.batch_size)
            break
        return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

# 
    # def test_dataloader(self):
        # Called when testing the model by calling: Trainer.test()
        # self.test_dataset = MyDataset(test_all.float(), sequence_len)
        # self.first_pred = 
        # return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
# 
# 
# -

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hs):
   
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
        out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size)     
        out = self.fc(out)
        
        return out, hs



# +
input_size = len(hostnames)
hidden_size = 200
num_layers = 2
output_size = len(hostnames)

model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.float()
# -

output_size




# +
#model with early stopping

early_stop_callback = EarlyStopping(
   monitor='val_loss_batch',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='max'
)

# Instantiate model
model = NeuralNet(input_size, hidden_size, num_layers, output_size,learning_rate=0.0005, 
                train_all=train_all, valid_all=valid_all, test_all=test_all, batch_size=1)


#init training with early stopping
trainer = pl.Trainer(callbacks=[early_stop_callback]) #,check_val_every_n_epoch=5

# Train
trainer.fit(model)
# -

#train and validation loss plot
train_loss = model.train_loss
valid_loss = model.valid_loss
plt.figure(figsize=[8., 6.])
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# +
#train now without monitoring val_loss

# Instantiate model
model = NeuralNet(input_size, hidden_size, num_layers, output_size,learning_rate=0.0005, 
                train_all=train_all, valid_all=valid_all, test_all=test_all, batch_size=1)
# # Init Trainer
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model)
# -

#train and validation loss plot
train_loss = model.train_loss
valid_loss = model.valid_loss
plt.figure(figsize=[8., 6.])
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# +
#model to find optimal learning_rate 

from torch.autograd import Variable

# Instantiate model
model = NeuralNet(input_size, hidden_size, num_layers, output_size,learning_rate=0.0005, 
                train_all=train_all, valid_all=valid_all, test_all=test_all, batch_size=1)

# Initialize trainer
trainer = pl.Trainer(max_epochs=50, 
                    check_val_every_n_epoch=10, 
                    precision=32,
                    )

# It is implemented the built-in function for finding the
# optimal learning rate. Source: https://arxiv.org/pdf/1506.01186.pdf
lr_finder = trainer.tuner.lr_find(model, 
                        min_lr=0.0005, 
                        max_lr=0.005,
                        mode='linear')

# Plots the optimal learning rate
fig = lr_finder.plot(suggest=True)
fig.show()
# -

# # Once everything is done, let's train the model
# trainer.fit(model)
# # Testing the model
res = trainer.test()
res

# +
#######pred 
fut_pred = len(test_all)

test_inputs = (train_all[-train_window:])
# print(test_inputs)

model.eval()

#  Get predictions on test data
hs = None
for i in range(fut_pred):
    #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
    seq = (test_inputs[-train_window:]).unsqueeze(0) 
    with torch.no_grad():
        test_preds, hs = model(seq.float(), hs)
         #get last prediction and concatenate with pred tensor
        test_inputs = torch.cat((test_inputs,test_preds[-1].unsqueeze(0)),0)



tests_fin = test_inputs[-len(test_all):]


# +
# # Init Neural Net model
# nn_ = NeuralNet(input_size, hidden_size, num_layers, output_size,learning_rate=0.0005, 
#                 train_all=train_all, valid_all=valid_all, test_all=test_all)

# # Init Trainer
# trainer = pl.Trainer(max_epochs=10)

# # Train
# trainer.fit(nn_, train_dataloader, val_dataloader)


# -

for x, y in train_dataloader:
    print(x.view(3,312).shape)
    break

for x,y in get_batches(train_all,window=25):
    print(x.shape)
    break


# +
def train(model, epochs, train_set, train_window, valid_data=None, lr=0.001, print_every=10):

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    valid_loss = []
    
    for e in range(epochs):
        print("EPOCH {} ****************************\n".format(e))
        hs = None
        t_loss = 0
        for x, y in get_batches(train_set, train_window):

            opt.zero_grad()
            
            # Create batch_size dimension
            x = x.unsqueeze(0)
            out, hs = model(x, hs)
            hs = tuple([h.data for h in hs])
            
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        
#             print("loss: ", loss, "loss.item(): ", loss.item())
#             print("t_loss: ",t_loss)
        
        v_loss = 0
        if valid_data is not None:
            model.eval()
            hs_val = hs
            for val_x, val_y in get_batches(valid_data, train_window):
                with torch.no_grad():
                    val_x = val_x.unsqueeze(0)
                    preds, hs_val = model(val_x, hs_val)
#                     hs_val = tuple([h.data for h in hs_val])
                    loss_val = criterion(preds, val_y)
                    v_loss += loss_val.item()
            
#                 print("loss_val: ", loss_val, " loss_val.item(): ", loss_val.item())
#                 print("v_loss: ",v_loss)
                

            valid_loss.append(np.mean(v_loss)) #mean of scalar is the scalar itself
            

            model.train()
            
        train_loss.append(np.mean(t_loss))#mean of scalar is the scalar itself

            
        if e % print_every == 0:
            print("valid_loss : ", valid_loss)
            print("train_loss : ", train_loss)

            print(f'Epoch {e}:\nTraining Loss: {train_loss[-1]}')
            if valid_data is not None:
                print(f'Validation Loss: {valid_loss[-1]}')
    
    plt.figure(figsize=[8., 6.])
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.show()

# +
# %%time

train_window = 25
epochs = 200
train(model, epochs, train_all.float(), train_window, valid_data=valid_all.float() ,lr=0.0005)

# +
fut_pred = len(test_all)

test_inputs = (train_all[-train_window:])
# print(test_inputs)

model.eval()

#  Get predictions on test data
hs = None
for i in range(fut_pred):
    #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
    seq = (test_inputs[-train_window:]).unsqueeze(0) 
    with torch.no_grad():
        test_preds, hs = model(seq.float(), hs)
         #get last prediction and concatenate with pred tensor
        test_inputs = torch.cat((test_inputs,test_preds[-1].unsqueeze(0)),0)



tests_fin = test_inputs[-len(test_all):]
# -

#transform predictions tensor to dataframe
tests_fin.shape # len(Test_all), num_machines
preds_df = pd.DataFrame(tests_fin.numpy())
preds_df_unscaled = preds_df.copy()
preds_df_unscaled

#undo predictions rescaling
preds_array_unscaled = scaler.inverse_transform(preds_df)
dic = {i:h for i,h in zip(np.arange(0,preds_df.columns.stop), hostnames)}
preds_df = preds_df.rename(columns=dic)
preds_array_unscaled

#replace predictions unscaled values in dataframe
preds_df.loc[:,] = preds_array_unscaled
preds_df


# +
# ### TEST IF reslcaling is working
# #original dataset
# m0 = dataset.iloc[:,0]
# sc = MinMaxScaler(feature_range=(-1, 1))
# #column sclaed individually
# m0_array_scaled = sc.fit_transform(m0.values.reshape(-1,1))
# print(m0)
# print(m0_array_scaled)

# #prediction
# preds_m0 = preds_df_unscaled[0].copy()
# print("\n scaled prediction")
# print(preds_m0)
# print("\n unscaled prediction after being uscaled with column scaler sc")
# print(sc.inverse_transform(preds_m0.values.reshape(-1, 1)))
# print("rescaling is working")

# +
def plot(hostnames,dataset,preds_df,len_train):
    fig,axs = plt.subplots(len(hostnames),1,figsize=(20,30),sharex=True, sharey=True)

    for ax,h in zip(axs, hostnames):
        
        #ground truth
        df_mach = dataset.loc[:, h].copy()
        df_mach = df_mach.iloc[:len_train+len(preds_df)]
#         gd_th_interp = savgol_filter(df_mach.values, 21, 3)
        gd_th_interp = df_mach.values
        x_gd_th = np.arange(0,len_train+len(preds_df),1)
        
        #predictions
#         i = np.where(hostnames==h)[0][0]
        pred = preds_df.loc[:,h]
        x_pred = np.arange(len_train, len_train+len(pred),1)
        
        #plot
        ax.plot(x_gd_th,gd_th_interp,  label='Ground Truth', )
        ax.plot(x_pred, pred, 'r--', label='Test Predictions')
        ax.set_title(h)
        
        ax.set_ylim([-10,75])
        
    fig.subplots_adjust(hspace = 0.2)
    
plot(hostnames[:10],dataset,preds_df,len(train_all))


# -

plot(hostnames[10:20],dataset,preds_df,len(train_all))

plot(hostnames[20:30],dataset,preds_df,len(train_all))

plot(hostnames[30:40],dataset,preds_df,len(train_all))




