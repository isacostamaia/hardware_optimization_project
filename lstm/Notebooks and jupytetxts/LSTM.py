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
# # %%cmd
# pip freeze > requirements.txt

# +
# # %%cmd
# pip install h5py

# +
# # %%cmd
# deactivate

# +
from datetime import datetime, timedelta


import numpy as np
import pandas as pd


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from pytorch_forecasting.metrics import MAPE
from tslearn.metrics import dtw
from torch.utils.data import Dataset, DataLoader
from soft_dtw_cuda import SoftDTW



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
#     'start_date': datetime.now() - timedelta(weeks =9),
    'start_date': datetime.fromisoformat('2021-03-25 11:00:00') , 
    # datetime
    'end_date': datetime.fromisoformat('2021-05-27 10:00:00'),
    # 'windows', 'linux'
    'os': 'windows',
    # List of host names
    'machines_to_include':None,
    # List of host names
    'machines_to_exclude': ['wip132dsy', 'ssdwip017dsy', 'ssdwip021dsy', 'ssdwip022dsy'],
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


# +
##delete after 
array = dataset.values

# for i in array:
#     for j, e in enumerate(i):
#         print(e, end='')
#         if j!=len(i)-1:
#             print(', ', end = '')
#     print()
# for i in range(len(dataset.columns)):
#     print("'{}'".format(dataset.columns[i]), end = '')
#     print(', ', end='')
array
# -

dataset.info()
dataset.index
dataset.loc[:, 'client14dsy'].values


# +
def get_batches(data, window):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window). 
    """

    L = len(data)
    for i in range(L - window):
        x_sequence = data[i:i + window]
        y_sequence = data[i+1: i + window + 1] 
        yield x_sequence, y_sequence;
        
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

# train_dataset = MyDataset(train_all.float(),q=train_window)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)

# valid_dataset = MyDataset(valid_all.float(),q=train_window)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)

# train_loader = get_batches_dataloader(train_all, train_window, batch_size = 1)
# valid_loader = get_batches_dataloader(valid_all, train_window, batch_size = 1)

# for x, y in (train_loader):
#     print(x, y)
#     print(x.shape, y.shape)


# -

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hs):
   
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
#         out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size)     
        out = self.fc(out)
        
        return out, hs


# +
input_size = len(hostnames)
hidden_size = 200
num_layers = 2
output_size = len(hostnames)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.float()
if device == 'cuda':
    model = model.to(device) 


# +
def train(model, epochs, train_set, train_window, device, valid_data=None, lr=0.001, batch_size =1, 
          print_every=10, loss_fct = 'MSE'):

    if device == 'cuda':
        model.cuda()
    
    if loss_fct == "MSE":
        criterion = nn.MSELoss()
    elif loss_fct == "MAPE":
        criterion = MAPE()
    elif loss_fct == "DTW":
        criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()

    
#appereantly MAPE cannot deal with 3d data to compute loss...do an average with each one of the samples?


    
    opt = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    valid_loss = []
    
    for e in range(epochs):
        print("EPOCH {} ****************************\n".format(e))
        hs = None
        t_loss = []
        for x, y in get_batches_dataloader(train_set, train_window, batch_size = batch_size):
            if device == 'cuda':
                x = x.cuda().to(device)
                y = y.cuda().to(device)

            
            opt.zero_grad()
            
            # Create batch_size dimension if it doesn't exists
            if (len(x.shape)==2):
                x = x.unsqueeze(0)
            out, hs = model(x, hs)
            hs = tuple([h.data for h in hs])
            
            #to compute mape loss
            if loss_fct == "MAPE":
                loss = torch.zeros(1)
                print("initial loss", loss)
                if device == 'cuda':
                    loss = loss.cuda().to(device) 
                for b in range(batch_size):
                    out_one_of_batch = out[b,:,:]
                    y_one_of_batch = y[b,:,:]
                    loss  = loss + criterion(out_one_of_batch, y_one_of_batch)
                loss = loss/batch_size

            else:
                loss = criterion(out, y)
                print(loss)
                if loss_fct == 'DTW':
                    loss = torch.mean(loss)
                    print("mean batch loss",loss)



            loss.backward()
            opt.step()
            t_loss.append(loss.item())

        v_loss = []
        if valid_data is not None:
            print(" \n Entered in validation \n")
            model.eval()
            hs_val = hs
            for val_x, val_y in get_batches_dataloader(valid_data, train_window, batch_size =batch_size):
                if device == 'cuda':
                    val_x = val_x.cuda().to(device)
                    val_y = val_y.cuda().to(device)
                
                with torch.no_grad():
                    if (len(val_x.shape)==2):
                        val_x = val_x.unsqueeze(0)
                    preds, hs_val = model(val_x, hs_val)
                    hs_val = tuple([h.data for h in hs_val])
                    
                    #to compute mape loss
                    if loss_fct == "MAPE":
                        loss_val = torch.zeros(1)
                        if device == 'cuda':
                            loss_val = loss_val.cuda().to(device) 
                        for b in range(batch_size):
                            
                            preds_one_of_batch = preds[b,:,:]
                            val_y_one_of_batch = val_y[b,:,:]
                            loss_val  = loss_val + criterion(preds_one_of_batch, val_y_one_of_batch)
                        loss_val = loss_val/batch_size

                        
                    else:
                        loss_val = criterion(preds, val_y)
                        print(loss)
                        if loss_fct == 'DTW':
                            loss = torch.mean(loss)
                            print("mean batch loss",loss)
                        
                    v_loss.append(loss_val.item())
            
                

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


#using data with batch size already included
train_window = 25
epochs = 200
train(model, epochs, train_all, train_window, device, valid_data=valid_all ,lr=0.0005, batch_size = 1, print_every=1)

# +
# torch.tensor([1,2,3]) + torch.tensor([1,2,3])
# criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
# criterion = MAPE()
# # criterion = nn.MSELoss()
# isinstance(criterion, nn.MSELoss)
# print(criterion)

# +
## Verifying how the losses behave
print("Behaviour of losses\n")

tms1 = dataset.iloc[:300,0].values
tms2 = dataset.iloc[:300,1].values
#shift
tms3 = np.roll(tms2,-200)

tms4 = dataset.iloc[:300,3].values

#transform into tensor of (seq_len, feature) Shape. (300,1)
tms1 = torch.tensor(tms1).unsqueeze(1)
tms2 = torch.tensor(tms2).unsqueeze(1)
tms3 = torch.tensor(tms3).unsqueeze(1)
tms4 = torch.tensor(tms4).unsqueeze(1)


plt.plot(tms1, label = "tms1")
plt.plot(tms2, label = "tms2")
plt.plot(tms3, label = "tms3")
plt.plot(tms4, label = "tms4")
plt.legend()

#DTW
criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
loss12 = criterion(tms1.unsqueeze(0), tms2.unsqueeze(0))
loss13 = criterion(tms1.unsqueeze(0), tms3.unsqueeze(0))
loss14 = criterion(tms1.unsqueeze(0), tms4.unsqueeze(0))

#MAPE
criterion_mape = MAPE()
loss12_mape = criterion_mape(tms1, tms2)
loss13_mape = criterion_mape(tms1, tms3)
loss14_mape = criterion_mape(tms1, tms4)

#MSE
criterion_mse = nn.MSELoss()
loss12_mse = criterion_mse(tms1, tms2)
loss13_mse = criterion_mse(tms1, tms3)
loss14_mse = criterion_mse(tms1, tms4)

print("\n *DTW loss*")
print("loss tms1 and tms2", loss12)
print("loss tms1 and tms3", loss13)
print("loss tms1 and tms4", loss14)

print("\n *MAPE loss*")
print("loss tms1 and tms2", loss12_mape)
print("loss tms1 and tms3", loss13_mape)
print("loss tms1 and tms4", loss14_mape)

print("\n *MSE loss*")
print("loss tms1 and tms2", loss12_mse)
print("loss tms1 and tms3", loss13_mse)
print("loss tms1 and tms4", loss14_mse)

valid_all.shape

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

test_all.shape
test_all[i:i+prediction_range].shape
test_inputs
preds.cuda().to(device)

# +
prediction_range = 3

# test_all = test_all.unsqueeze(0)

# .unsqueeze(0)

fut_pred = len(test_all)

test_inputs = (train_all[-train_window:])
print("shape test_inputs", test_inputs.shape)
print("shape test_all", test_all.shape)
#preds: ephemeral tensor whose len reaches till prediction_range
#all_preds: tensor that contains all predicitons
#size of a single prediction
preds = torch.tensor([])
all_preds = torch.tensor([])

if device == 'cuda':
    test_inputs = test_inputs.cuda().to(device)
    preds = preds.cuda().to(device)
    all_preds = all_preds.cuda().to(device)
    test_all = test_all.cuda().to(device)
    
model.eval()

#  Get predictions on test data
hs = None
for i in range(fut_pred):
    
    #get last predictions by getting values in preds tensor and the number of ground truth values needed to complete train_window
    #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
    seq = torch.cat((test_inputs[-(train_window - preds.shape[0]):], preds),0).unsqueeze(0).float()
    
    if device == 'cuda':
        seq = seq.cuda().to(device)
        
    with torch.no_grad():
        test_preds, hs = model(seq, hs)
        #if it is three dimensional, take out the first dimension (that will be equal 1, the batch_size)
        test_preds = test_preds.squeeze()
        
        #get last value (prediction) and put in preds vector
        preds = torch.cat((preds, test_preds[-1].unsqueeze(0)),0).cuda().to(device)
        #concatenate in all predictions tensor
        all_preds = torch.cat((all_preds, test_preds[-1].unsqueeze(0)),0)
        
        #preds: tensor with predictions used to make more predictions. Its first dimension len gets till predictio range
        #test_inputs: tensor with ground truth values used to make predictions
        if (preds.shape[0]==prediction_range):
            preds = torch.tensor([]).cuda().to(device)
            test_inputs = torch.cat((test_inputs,test_all[i:i+prediction_range]),0)
        



# tests_fin = test_inputs[-len(test_all):]
tests_fin = all_preds
# -

#transform predictions tensor to dataframe
tests_fin.shape # len(Test_all), num_machines
preds_df = pd.DataFrame(tests_fin.cpu().numpy())
preds_df_unscaled = preds_df.copy()
preds_df_unscaled

#undo predictions rescaling
preds_array_unscaled = scaler.inverse_transform(preds_df)
dic = {i:h for i,h in zip(np.arange(0,preds_df.columns.stop), hostnames)}
preds_df = preds_df.rename(columns=dic)
preds_array_unscaled

#replace predictions unscaled values in dataframe
preds_df= pd.DataFrame(preds_array_unscaled, columns=hostnames)
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

plot(hostnames[40:50],dataset,preds_df,len(train_all))

plot(hostnames[50:60],dataset,preds_df,len(train_all))

plot(hostnames[270:280],dataset,preds_df,len(train_all))

criterion = SoftDTW(gamma=1.0, use_cuda=True)

print(criterion)

criterion.gamma


