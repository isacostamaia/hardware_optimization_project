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
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
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
    'start_date': datetime.fromisoformat('2021-03-25 11:00:00'), #None , 
    # datetime
    'end_date': datetime.fromisoformat('2021-05-27 10:00:00'), #,None
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
# -


dataset.info()
dataset.index
dataset.loc[:, 'client14dsy'].values

dataset_scaled.values


# ## Encoding

# +
def encode(x):
    if x == 0:
        return np.array([1.,0.,0.])
    elif x == 1:
        return np.array([1.,1.,0.])
    elif x == 2:
        return np.array([1.,1.,1.])
    
def decode(x):
    if (x == np.array([1.,0.,0.])).all():
        return 0
    elif (x == np.array([1.,1.,0.])).all():
        return 1
    elif (x == np.array([1.,1.,1.])).all():
        return 2
    else:
        return 3


# +
#find categories
num_classes = 3
rows, num_hosts = dataset.values.shape
X = dataset.values.reshape(-1,1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est.fit(X)
Xt = est.transform(X)
categorical_dataset = Xt.reshape(-1, num_hosts)
dataset_categorical  =  pd.DataFrame(categorical_dataset, columns = dataset.columns)

#apply encoding
dataset_categorical_encoded = dataset_categorical.applymap(encode)

#find train test valid tensor of encoded values
tensor_dataset_categorical = torch.tensor(np.vstack( dataset_categorical_encoded.values.ravel() ).reshape(-1,num_hosts,num_classes))
train_all_encoded, test_all_encoded, valid_all_encoded = split_data(tensor_dataset_categorical)


#build dataloader 
# -

print("shape de cada exemplo y antes ", train_all[0].shape)
print("shape de cada exemplo y depois de encoded ", train_all_encoded[0].shape)

# +
train_window = 25
num_timestep_prev = 3
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
    def __init__(self, data, data_categorical, q, num_timestep_prev = num_timestep_prev):
        self.data = torch.Tensor(data)
        self.data_categorical = torch.Tensor(data_categorical)
        self.window = q
        self.num_timestep_prev = num_timestep_prev

    def __len__(self):
        return len(self.data) -  self.window - self.num_timestep_prev + 1

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
#         print("dataloader:  x shape ", x.shape)
        y = self.data_categorical[index + self.window:index + self.window + self.num_timestep_prev] #take a single position for y (the postion that follows x)
#         print("dataloader: y shape ", y.shape)
        return x , y
    
def get_batches_dataloader(data,data_encoded, window, batch_size):
    dataset = MyDataset(data.float(), data_encoded.float(),q = window)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last=True)

train_dataset = MyDataset(train_all.float(),train_all_encoded.float(),q=train_window)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)

# valid_dataset = MyDataset(valid_all.float(),q=train_window)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)

# train_loader = get_batches_dataloader(train_all, train_window, batch_size = 1)
# valid_loader = get_batches_dataloader(valid_all, train_window, batch_size = 1)

for x, y in (train_loader):
#     pass
    print("x: ")
    print(x)
    print("x shape = ",x.shape)
    print("y: ")
    print(y)
    print("y shape = ",y.shape)
    break
# -

# ###### little toy case

# +
X_dull = np.array([[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16], [9,10,11,12,13,14,15,16], [9,10,11,12,13,14,15,16]]).T
print("Initial X\n",pd.DataFrame(X_dull))

X_dull_res = X_dull.reshape(-1,1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est.fit(X_dull_res)
Xt = est.transform(X_dull_res)
Xt
X_categorical_dataset = Xt.reshape(-1, X_dull.shape[1])
X_dataset_categorical  =  pd.DataFrame(X_categorical_dataset)

print("\nCategorical X\n",X_dataset_categorical)

X_dataset_categorical_encoded = X_dataset_categorical.applymap(encode)

print("\nCategorical X Encoded \n",X_dataset_categorical_encoded)
# -

X_tensor_dataset_categorical = torch.tensor(np.vstack(X_dataset_categorical_encoded.values.ravel() ).reshape(-1,4,num_classes))
#4 would mean the number of machines
X_train_loader = get_batches_dataloader(torch.tensor(X_dull),X_tensor_dataset_categorical, window=3, batch_size=1)

i = 0
y_list = []
x_list = []
for x, y in (X_train_loader):
#     pass
    y_list.append(y)
    x_list.append(x)
    print("x: ")
    print(x)
    print("x shape = ",x.shape)
    print("y: ")
    print(y)
    print("y shape = ",y.shape)
    i+=1
#     if i==:
#         break

criterion = nn.MSELoss(reduction="sum")
criterion(y,y)

# +
# um = np.vstack(  dataset_categorical_encoded.iloc[0,:].values )
# dois = np.vstack(  dataset_categorical_encoded.iloc[-1,:].values )
# criterion = nn.MSELoss(reduction="sum")
# loss = criterion(torch.tensor(um), torch.tensor(dois))
# loss

# +
#TEST if we get y with multiple time positions

criterion = nn.MSELoss(reduction="sum")

np.vstack(dataset_categorical_encoded.iloc[5:7,:].values.ravel() )

torch.tensor(np.vstack(  dataset_categorical_encoded.iloc[5:7,:].values.ravel() ))

criterion(torch.tensor(np.vstack(dataset_categorical_encoded.iloc[5:7,:].values.ravel() )), torch.tensor(np.vstack(dataset_categorical_encoded.iloc[5:7,:].values.ravel() )))
# -

torch.tensor(np.vstack(  dataset_categorical_encoded.iloc[5:7,:].values.ravel() )).shape


# ###### end little toy case

# +
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size*train_window, output_size*num_classes*num_timestep_prev)
        
    def forward(self, x, hs):
   
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
    
#         print("out apos lstm e antes da fc layer", out.shape)

#         out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size) 
        #condensate all timesteps so we get a single position at the end
        out = out.reshape(1, train_window*self.hidden_size)
        
#         print("out apos lstm e antes da fc layer reshaped ", out.shape)

        out = self.fc(out)
    
#         print("out saindo da fc layer", out.shape)
        
        out = out.reshape(-1, num_timestep_prev, num_hosts, num_classes) #shape([1, 3, 309, 3])
        #since we modified y in the daatloader to retrieev only one time step position, we modify out as the follow.
        #if we xant to keep y with more positions we delete the following line and we change the dataloader.
        
#         print("out saindo da fc layer reshaped", out.shape)

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
def train(model, epochs, train_set, train_set_encoded, train_window, device, 
          valid_data=None, valid_data_encoded = None,lr=0.001, batch_size =1, 
          print_every=10, loss_fct = 'MSE'):

    if device == 'cuda':
        model.cuda()
    
    if loss_fct == "MSE":
        criterion = nn.MSELoss(reduction="sum")
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
        for x, y in get_batches_dataloader(train_set, train_set_encoded, train_window, batch_size = batch_size):
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
#                 print("loss ", loss)
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
            for val_x, val_y in get_batches_dataloader(valid_data, valid_data_encoded, train_window, batch_size =batch_size):
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
#                         print(loss)
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
train(model, epochs, train_all, train_all_encoded, train_window, device, valid_data=valid_all, valid_data_encoded = valid_all_encoded ,lr=0.0005, batch_size = 1, print_every=1)

# +
import math
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
for i in range(math.ceil(fut_pred/prediction_range)):
    
    #get last predictions by getting values in preds tensor and the number of ground truth values needed to complete train_window
    #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
    print("test_inputs[-train_window :].shape ", test_inputs[-train_window :].shape)
    seq = test_inputs[-train_window :].unsqueeze(0).float()
    print("seq shape ", seq.shape)
    if device == 'cuda':
        seq = seq.cuda().to(device)
        
    with torch.no_grad():
        test_preds, hs = model(seq, hs)
        print("test_preds.shape", test_preds.shape)
        #if it is three dimensional, take out the first dimension (that will be equal 1, the batch_size)
        test_preds = test_preds.squeeze()
        print("test_preds.squeeze().shape", test_preds.shape)
        #concatenate in all predictions tensor
        all_preds = torch.cat((all_preds, test_preds),0)
        #test_inputs: tensor with ground truth values used to make predictions
        #update tensor with ground truth values
        test_inputs = torch.cat((test_inputs,test_all[i:i+prediction_range]),0)
        



# tests_fin = test_inputs[-len(test_all):]
tests_fin = all_preds[:len(test_all)]

#check size
# tests_fin.shape
# test_all.shape
# -

tests_fin

a = torch.round(tests_fin).cpu().numpy()
a

torch.round(tests_fin)
a = torch.round(tests_fin).cpu().numpy()
tests_fin_category = [[decode(el) for el in codes_list] for  codes_list in a]
tests_predict = pd.DataFrame(np.array(tests_fin_category), columns = dataset.columns)
#predicted values in test
tests_predict

#groundtruth values
dataset_categorical

a.shape

#ground truth values in test
tests_ground_truth = dataset_categorical.iloc[train_all_encoded.shape[0]:train_all_encoded.shape[0]+len(tests_fin_category)]
tests_ground_truth

# +
#confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(tests_ground_truth.values.ravel(), tests_predict.values.ravel(), normalize = 'true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=[0,1,2,3])
disp.plot()

# +
#end

# +
#Didnt get the meaning of sigmoid

# m = nn.Sigmoid()
# input = tests_fin
# output = m(input)
# output
# +
#visualizations

# +
#ground truth values

all_alias = dataset.columns

def color_category(cat):
    if cat==0:
        return 'g'
    if cat ==1:
        return 'y'
    if cat==2:
        return 'r'
    return'c'

def plot_activity_class(some_alias, dataset, dataset_categorical):
    fig,axs = plt.subplots(len(some_alias),1,figsize=(20,20),sharex=True, sharey=True)
    for ax,alias in zip(axs,some_alias):
        ts_cat = dataset_categorical.loc[:, alias]
        ts = dataset.loc[:, alias]
        ax.plot(ts,label=alias, c = 'black')
        colors_categories = [color_category(el) for el in ts_cat.values]
        ax.scatter(ts.index, ts.values,  c = colors_categories ,label=alias, s=200)
        ax.set_ylim([0,75])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.8)

plot_activity_class(all_alias[:int(len(all_alias)/30)], dataset, dataset_categorical)    

# +
cpu_values = dataset.values.ravel()
cpu_categories = dataset_categorical.values.ravel()



freq, bins, patches = plt.hist(cpu_categories, bins= [-0.5,0.5,0.5,1.5,1.5,2.5,])


# x coordinate for labels
bin_centers = np.diff(bins)*0.5 + bins[:-1]

n = 0
for fr, x, patch in zip(freq, bin_centers, patches):
  height = int(freq[n])
  plt.annotate("{}".format(height),
               xy = (x, height),             # top left corner of the histogram bar
               xytext = (0,0.2),             # offsetting label position above its bar
               textcoords = "offset points", # Offset (in points) from the *xy* value
               ha = 'center', va = 'bottom'
               )
  n = n+1

plt.title("Classes distribution (all ground truth dataset) found using kmean-clustering with k=3")
plt.show;
# -

cpu_tuple = cpu_values[cpu_categories==0], cpu_values[cpu_categories==1], cpu_values[cpu_categories==2]
plt.eventplot(cpu_tuple, colors = ['g', 'y', 'r'])
plt.title("Distribution of each one of the classes")
plt.xlabel("CPU percentage")
plt.yticks(color='None')

# +
# tests_ground_truth

# def plot_activity_class(some_alias, ground_truth_categorical, tests_categorical):
#     fig,axs = plt.subplots(len(some_alias),1,figsize=(20,20),sharex=True, sharey=True)
#     for ax,alias in zip(axs,some_alias):
#         ts_gt_cat = ground_truth_categorical.loc[:, alias]
#         colors_ts_gt_cat = [color_category(el) for el in ts_gt_cat.values]
#         ts_test_cat = tests_categorical.loc[:,alias]
#         colors_ts_test_cat = [color_category(el) for el in  ts_test_cat.values]
#         ts = dataset.loc[:, alias]
#         ax.scatter(ts_test_cat.index, ts_gt_cat.values,  c = colors_ts_gt_cat ,label=alias, s=100)
#         ax.scatter(ts_test_cat.index, ts_test_cat.values,  c = colors_ts_test_cat ,label=alias, s=100, marker = '*', edgecolors = 'b')
#         ax.set_ylim([0,4])
#         ax.set_title(alias)

#     fig.subplots_adjust(hspace = 0.8)

# plot_activity_class(all_alias[:int(len(all_alias)/30)], tests_ground_truth, tests_predict)    


def plot_activity_class(some_alias, ground_truth_categorical, tests_categorical):
    fig,axs = plt.subplots(len(some_alias),1,figsize=(20,30),sharex=True, sharey=True)
    for ax,alias in zip(axs,some_alias):
        ts_gt_cat = ground_truth_categorical.loc[:, alias]
        colors_ts_gt_cat = [color_category(el) for el in ts_gt_cat.values]
        ts_test_cat = tests_categorical.loc[:,alias]
        colors_ts_test_cat = [color_category(el) for el in  ts_test_cat.values]
        ts = dataset.loc[:, alias]
        ax.scatter(ts_test_cat.index,['groud truth']*len(ts_test_cat.index),  c = colors_ts_gt_cat ,label=alias)
        ax.scatter(ts_test_cat.index, ['predicted']*len(ts_test_cat.index),  c = colors_ts_test_cat ,label=alias, marker = '*')
        ax.set_ylim([0,1.2])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.9)

plot_activity_class(all_alias[:int(len(all_alias)/30)], tests_ground_truth, tests_predict)   

# +
#instead of plotting class 3, that do not exists, we''l replace by Nan
replace_nan = lambda x: None if x == 3 else x


def plot_activity_class(some_alias, ground_truth_categorical, tests_categorical):
    fig,axs = plt.subplots(len(some_alias),1,figsize=(20,30),sharex=True, sharey=True)
    
    #replace class 3 by nan
    tests_categorical = tests_predict.applymap(replace_nan)
    
    for ax,alias in zip(axs,some_alias):
        ts_gt_cat = ground_truth_categorical.loc[:, alias]
        colors_ts_gt_cat = [color_category(el) for el in ts_gt_cat.values]
        ts_test_cat = tests_categorical.loc[:,alias]
        colors_ts_test_cat = [color_category(el) for el in  ts_test_cat.values]
        ts = dataset.loc[:, alias]
        ax.plot(ts_test_cat.index, ts_gt_cat, label=alias, c = 'black')
        ax.plot(ts_test_cat.index, ts_test_cat,'r--',label=alias)
        ax.set_ylim([0,3.5])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.9)

plot_activity_class(all_alias[:int(len(all_alias)/30)], tests_ground_truth, tests_predict)   

# +
import chardet
import pandas as pd

with open('Machines_Suivi.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


# pd.read_csv('filename.csv', encoding=result['encoding'])

machines_df = pd.read_csv('Machines_Suivi.csv', sep=';',  encoding=result['encoding'])

machines_df

# +
machines_df.loc[:, 'Name'] = [x.lower() for x in machines_df.Name]
machines_df = machines_df[['Name', 'Projet', 'Role']]

machines_df
# -

right_predict_per = [(tests_predict==tests_ground_truth.reset_index(drop=True))[alias].values.sum()/len(tests_predict) for alias in all_alias]

# +
# # ri[((tests_predict==tests_ground_truth.reset_index(drop=True)) & tests_ground_truth ==0)[alias].values.sum()/len(tests_predict) for alias in all_alias]

# ((tests_predict==tests_ground_truth.reset_index(drop=True)) & tests_ground_truth ==1)['client14dsy']

# # (tests_predict==tests_ground_truth.reset_index(drop=True))['client14dsy']
# -

results_df = pd.DataFrame({"Name": all_alias, "Accuracy": right_predict_per})
results_df

df = results_df.copy().merge(
    machines_df,
    how='left',
    on='Name'
)
df['Projet_Role'] =  df['Projet'] + df["Role"]

df.sort_values(by = 'Accuracy').head(40)

df.sort_values(by = 'Accuracy').head(40).Projet.value_counts()

df.sort_values(by = 'Accuracy').head(40).Role.value_counts()

df.sort_values(by = 'Accuracy').head(40).Projet_Role.value_counts()

df.sort_values(by = 'Accuracy').tail(30)

df.groupby('Projet_Role').agg(mean_accuracy = ('Accuracy',np.mean), std_acccuracy = ('Accuracy', np.std)).sort_values(by='mean_accuracy')

df.groupby('Projet').agg(mean_accuracy = ('Accuracy',np.mean), std_acccuracy = ('Accuracy', np.std)).sort_values(by='mean_accuracy')

df.groupby('Role').agg(mean_accuracy = ('Accuracy',np.mean), std_acccuracy = ('Accuracy', np.std)).sort_values(by='mean_accuracy')

# +
# import seaborn as sns
# sns.boxplot(x = "Projet_Role", y = 'Accuracy', data = df)
# plt.xticks(rotation=45)
# -

df.groupby('Projet_Role').agg(mean_accuracy = ('Accuracy',np.mean), std_acccuracy = ('Accuracy', np.std)).sort_values(by='mean_accuracy', ascending = False)


# +
def boxplot_sorted(df, by, column):
  df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
  meds = df2.mean().sort_values()
  df2[meds.index].boxplot(rot=90)

boxplot_sorted(df, by=['Projet_Role'], column="Accuracy")
# -

df.groupby('Projet_Role').agg(mean_accuracy = ('Accuracy',np.mean), std_acccuracy = ('Accuracy', np.std), num_machines = ('Accuracy', len)).sort_values(by='mean_accuracy', ascending = False)

len(df[df.Projet_Role == 'RelOpsBuild Win'])

relops_odt_replay = df[df.Projet_Role == 'RelOpsODT Replay'].Name.to_list()

plot_activity_class(relops_odt_replay, tests_ground_truth, tests_predict) 


