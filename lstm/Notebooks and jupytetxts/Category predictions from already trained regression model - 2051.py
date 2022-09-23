# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from datetime import datetime, timedelta

import chardet
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df, hosts_freqseries
from data_source.dataframe import hosts_timeseries
# -

# # Load already trained model

# +
PATH = "some_nice_models/2051_loop2_ssdwip016dsy/checkpoints/epoch=249-step=263499.ckpt"

model_ = torch.load(PATH)

# input_size = model_['hyper_parameters']['n_features']
# hidden_size = model_['hyper_parameters']['hidden_size']
# num_layers = model_['hyper_parameters']['num_layers']

# model = LSTM(input_size, hidden_size, num_layers, input_size)
# model.load_state_dict(model_['state_dict'])
# model.eval()
# -

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
                 start_date):
#                  , 
#                  end_date, 
#                  num_weeks):

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
#         self.num_weeks_train = num_weeks
#         self.end_date = end_date#

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

# +
model = LSTMRegressor.load_from_checkpoint(PATH)
model = model.float()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if device == 'cuda':
    model = model.to(device) 
# -

train_window = model_['hyper_parameters']['seq_len']
train_window

model_['hyper_parameters']['hidden_size']
model_['hyper_parameters']['learning_rate']

model_

# # Retrieve data from db

machines_model = ['client14dsy', 'client14xdsy', 'cmibuild1dsy', 'dell1843dsy',
       'dell1883dsy', 'dell1963dsy', 'dell3353dsy', 'dell3354dsy',
       'dell3355dsy', 'dell3356dsy', 'dell3357dsy', 'dell3358dsy',
       'dell3359dsy', 'dell3360dsy', 'dell3361dsy', 'dell3364dsy',
       'dell3365dsy', 'dell3366dsy', 'dell3367dsy', 'dell3368dsy',
       'dell3369dsy', 'dell3370dsy', 'eno24adsy', 'eno24bdsy',
       'eno24cdsy', 'eno24dsy', 'enojarnacdsy', 'enoprj2016dsy',
       'enowinb001dsy', 'enowinb002dsy', 'enowinb003dsy', 'enowinb004dsy',
       'enowip01dsy', 'enowip02dsy', 'enowip03dsy', 'enowip04dsy',
       'enowip05dsy', 'enowip07dsy', 'enowip08dsy', 'enowip09dsy',
       'enowip10dsy', 'enowip11dsy', 'enowip12dsy', 'enowip13dsy',
       'enowip14dsy', 'enowip15dsy', 'infra14xdsy', 'infra15dsy',
       'pool08dsy', 'pool09dsy', 'pool10dsy', 'pool11dsy', 'pool12dsy',
       'pool13dsy', 'pool14dsy', 'pool15dsy', 'pool23dsy', 'pool24dsy',
       'pool26dsy', 'pool27dsy', 'pool28dsy', 'pool29dsy', 'pool30dsy',
       'pool31dsy', 'pool32dsy', 'pool33dsy', 'pool34dsy', 'pool35dsy',
       'pool36dsy', 'pool37dsy', 'pool38dsy', 'pool39dsy', 'pool40dsy',
       'pool43dsy', 'pool44dsy', 'pool45dsy', 'pool46dsy', 'pool47dsy',
       'pool48dsy', 'pool49dsy', 'pool50dsy', 'pool51dsy', 'pool52dsy',
       'pool53dsy', 'pool55dsy', 'pool56dsy', 'pool59dsy', 'pool60dsy',
       'preint2017xdsy', 'ssdclient15xdsy', 'ssdhwip001dsy',
       'ssdhwip002dsy', 'ssdhwip003dsy', 'ssdhwip004dsy', 'ssdhwip005dsy',
       'ssdhwip006dsy', 'ssdhwip007dsy', 'ssdhwip008dsy', 'ssdhwip009dsy',
       'ssdhwip010dsy', 'ssdhwip011dsy', 'ssdhwip012dsy', 'ssdhwip013dsy',
       'ssdhwip014dsy', 'ssdhwip015dsy', 'ssdhwip016dsy', 'ssdhwip017dsy',
       'ssdhwip018dsy', 'ssdhwip019dsy', 'ssdhwip020dsy', 'ssdhwip021dsy',
       'ssdhwip022dsy', 'ssdhwip023dsy', 'ssdhwip024dsy', 'ssdhwip025dsy',
       'ssdinfra15xdsy', 'ssdprj2017xdsy', 'ssdwip001dsy', 'ssdwip002dsy',
       'ssdwip003dsy', 'ssdwip004dsy', 'ssdwip005dsy', 'ssdwip007dsy',
       'ssdwip008dsy', 'ssdwip009dsy', 'ssdwip020dsy', 'ssdwip023dsy',
       'ssdwip024dsy', 'vmyme010dsy', 'wip093dsy', 'wip094dsy',
       'wip095dsy', 'wip096dsy', 'wip103dsy', 'wip106dsy', 'wip107dsy',
       'wip112dsy', 'wip113dsy', 'wip114dsy', 'wip115dsy', 'wip117dsy',
       'wip118dsy', 'wip121dsy', 'wip122dsy', 'wip123dsy', 'wip125dsy',
       'wip126dsy', 'wip128dsy', 'wip131dsy', 'wip133dsy', 'wip134dsy',
       'wip135dsy', 'wip136dsy', 'wip137dsy', 'wip138dsy', 'wip139dsy',
       'wip141dsy', 'wip142dsy', 'wip146dsy', 'wip147dsy', 'wip148dsy',
       'wip149dsy', 'wip150dsy', 'wip152dsy', 'wip153dsy', 'wip154dsy',
       'wip155dsy', 'wip156dsy', 'wip160dsy', 'wip161dsy', 'wip163dsy',
       'wip164dsy', 'wip165dsy', 'wip166dsy', 'wip167dsy', 'wip168dsy',
       'wip170dsy', 'wip173dsy', 'wip174dsy', 'wip176dsy', 'wip177dsy',
       'wip178dsy', 'wip179dsy', 'wip181dsy', 'wip182dsy', 'wip183dsy',
       'wip184dsy', 'wip185dsy', 'wip186dsy', 'wip187dsy', 'wip188dsy',
       'wip192dsy', 'wip193dsy', 'wip196dsy', 'wip197dsy', 'wip198dsy',
       'wip199dsy', 'wip200dsy', 'wip201dsy', 'wip203dsy', 'wip204dsy',
       'wip205dsy', 'wip207dsy', 'wip208dsy', 'wip212dsy', 'wip213dsy',
       'wip214dsy', 'wip215dsy', 'wip216dsy', 'wip219dsy', 'wip220dsy',
       'wip222dsy', 'wip230dsy', 'wip233dsy', 'wip234dsy', 'wip237dsy',
       'wip238dsy', 'wip239dsy', 'wip241dsy', 'wip242dsy', 'wip243dsy',
       'wip244dsy', 'wip245dsy', 'wip246dsy', 'wip247dsy', 'wip248dsy',
       'wip249dsy', 'wip250dsy', 'wip251dsy', 'wip254dsy', 'wip255dsy',
       'wip256dsy', 'wip257dsy', 'wip258dsy', 'wip259dsy', 'wip261dsy',
       'wip262dsy', 'wip263dsy', 'wip264dsy', 'wip266dsy', 'wip267dsy',
       'wip270dsy', 'wip277dsy', 'wip278dsy', 'wip279dsy', 'wip284dsy',
       'wip285dsy', 'wip300dsy', 'wip302dsy', 'wip303dsy', 'wip305dsy',
       'wip310dsy', 'wip317dsy', 'wip318dsy', 'wip319dsy', 'wip320dsy',
       'wip322dsy', 'wip326dsy', 'wip327dsy', 'wip328dsy', 'wip329dsy',
       'wip330dsy', 'wip331dsy', 'wip332dsy', 'wip333dsy', 'wip334dsy',
       'wip335dsy', 'wip336dsy', 'wip337dsy', 'wip338dsy', 'wip339dsy',
       'wip340dsy', 'wip342dsy', 'wip343dsy', 'wip344dsy', 'wip345dsy',
       'wip346dsy', 'wip347dsy', 'wip348dsy', 'wip350dsy', 'wip351dsy',
       'wip352dsy', 'wip353dsy', 'wip354dsy', 'wip355dsy', 'wip356dsy',
       'wip357dsy', 'wip358dsy', 'wip359dsy', 'wip360dsy', 'wip361dsy',
       'wip363dsy', 'wip370dsy', 'wip124dsy', 'wip265dsy', 'wip316dsy',
       'wip210dsy', 'wip311dsy', 'wip312dsy', 'wip313dsy', 'wip314dsy',
       'wip315dsy', 'wip102dsy', 'wip097dsy', 'wip323dsy', 'wip189dsy',
       'enowip06dsy', 'pool05dsy', 'pool06dsy']

# +
# %%time


# Set interval & filters
query_params = {
    # 'day', 'hour', 'minute'
    'interval': 'hour',
    # datetime
#     'start_date': datetime.now() - timedelta(weeks = 8),
    'start_date': datetime(2021, 3, 25, 11, 0),
    # datetime
    'end_date': datetime(2021, 3, 25, 11, 0) + timedelta(weeks = 9),
#     'end_date': None,
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

df = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

print("number of machines of my dataset? ",len(df.hostname.unique()) == model_['hyper_parameters']['n_features'])

# # Rescale and Treat input data

# +
#interpolate values and get same date intervals for all
dfs, hostnames =  hosts_timeseries(df)
dfs = [a.rename(h) for a,h in zip(dfs,hostnames)]
dataset = pd.concat(dfs,axis=1)

#when using limit date of datetime(2021, 7, 13, 18, 13, 29, 150011
# #take only the last 20% because what comes before was used for training the model
# twenty_per_len = int(.2*len(dataset))
# dataset = dataset.iloc[-twenty_per_len:]

#take only the last 50% because why not
twenty_per_len = int(.5*len(dataset))
dataset = dataset.iloc[-twenty_per_len:]

#scale values
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset_array_scaled = scaler.fit_transform(dataset)
dataset_scaled = dataset.copy()
dataset_scaled.loc[:,:] = dataset_array_scaled


tensor_dataset = torch.tensor(dataset_scaled.values)
# -

test_inputs = tensor_dataset[:train_window]
test_all = tensor_dataset[train_window:]

# # Make predictions

# +
prediction_range = 3


fut_pred = len(test_all)

# test_inputs = (train_all[-train_window:])
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
        test_preds = model(seq)
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

# # Treat predictions 

# +
def treat_preds(tests_fin):
    #transform predictions tensor to dataframe
    tests_fin.shape # len(Test_all), num_machines
    preds_df = pd.DataFrame(tests_fin.cpu().numpy())


    #undo predictions rescaling
    preds_array_unscaled = scaler.inverse_transform(preds_df)
    dic = {i:h for i,h in zip(np.arange(0,preds_df.columns.stop), hostnames)}
    preds_df = preds_df.rename(columns=dic)


    #replace predictions unscaled values in dataframe
    preds_df= pd.DataFrame(preds_array_unscaled, columns=hostnames)
    return preds_df

preds_df = treat_preds(tests_fin)


# -

# # Plot predictions

# +
def plot(hostnames,dataset,preds_df,len_test):
    fig,axs = plt.subplots(len(hostnames),1,figsize=(20,30),sharex=True, sharey=True)

    for ax,h in zip(axs, hostnames):
        
        #ground truth
        df_gdth = dataset.loc[:, h].copy()
        df_gdth = df_gdth[-len_test:]

        #predictions
        df_pred = preds_df.loc[:,h]
        
        #plot
        ax.plot(df_pred.index,df_gdth.values,  label='Ground Truth', )
        ax.plot(df_pred.index, df_pred.values, 'r--', label='Test Predictions')
        ax.set_title(h)
        
        ax.set_ylim([-10,75])
        
    fig.subplots_adjust(hspace = 0.2)
    
plot(hostnames[:10],dataset,preds_df,len(test_all))
# -

plot(hostnames[30:40],dataset,preds_df,len(test_all))

# # Classify Predictions

# +
#categorize predictions
preds_res = preds_df.values.reshape(-1,1)
est_preds = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est_preds.fit(preds_res)
preds_res_cat = est_preds.transform(preds_res)
preds_res_cat
preds_cat = preds_res_cat.reshape(-1, preds_df.values.shape[1])
preds_df_cat  =  pd.DataFrame(preds_cat, columns = preds_df.columns)

#categorize ground truth in test
gdth_test_df = dataset.iloc[-len(test_all):, :]
gdth_res = gdth_test_df.values.reshape(-1,1)
est_gdth = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est_gdth.fit(gdth_res)
gdth_res_cat = est_gdth.transform(gdth_res)
gdth_res_cat
gdth_cat = gdth_res_cat.reshape(-1, gdth_test_df.values.shape[1])
gdth_df_cat  =  pd.DataFrame(gdth_cat, columns = gdth_test_df.columns)


# -

#confusion matrix
cm = confusion_matrix(gdth_df_cat.values.ravel(), preds_df_cat.values.ravel(), normalize = 'true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=[0,1,2])
disp.plot()


# +
def color_category(cat):
    if cat==0:
        return 'g'
    if cat ==1:
        return 'y'
    if cat==2:
        return 'r'
    return'c'

def plot_activity_class(some_alias, ground_truth_categorical, tests_categorical):
    '''
        some_alias: list with alias names
        ground_truth_categorical: dataframe with ground truth categories
        tests_categorical: dataframe with test categories
    '''
    
    fig,axs = plt.subplots(len(some_alias),1,figsize=(20,30),sharex=True, sharey=True)
    
    for ax,alias in zip(axs,some_alias):
        ts_gt_cat = ground_truth_categorical.loc[:, alias]
        colors_ts_gt_cat = [color_category(el) for el in ts_gt_cat.values]
        ts_test_cat = tests_categorical.loc[:,alias]
        colors_ts_test_cat = [color_category(el) for el in  ts_test_cat.values]
        ts = dataset.loc[:, alias]
        ax.plot(ts_test_cat.index, ts_gt_cat, label=alias, c = 'black')
        ax.plot(ts_test_cat.index, ts_test_cat,'r--',label=alias)
        ax.set_ylim([0,2.5])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.9)

plot_activity_class(hostnames[:int(len(hostnames)/30)], gdth_df_cat, preds_df_cat)   
# -

plot_activity_class(hostnames[30:40], gdth_df_cat, preds_df_cat)   

# # Analysing Roles

# +
with open('Machines_Suivi.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


# pd.read_csv('filename.csv', encoding=result['encoding'])

machines_df = pd.read_csv('Machines_Suivi.csv', sep=';',  encoding=result['encoding'])

machines_df.loc[:, 'Name'] = [x.lower() for x in machines_df.Name]
machines_df = machines_df[['Name', 'Projet', 'Role']]

# +
right_predict_per = [(preds_df_cat[alias] == gdth_df_cat[alias]).sum()/len(preds_df_cat) for alias in hostnames]

results_df = pd.DataFrame({"Name": hostnames, "Accuracy": right_predict_per})

df = results_df.copy().merge(
    machines_df,
    how='left',
    on='Name'
)

df['Projet_Role'] =  df['Projet'] + df["Role"]

df.head()
# -

# df.Role.unique()
len(df[(df.Role == 'Build Win') & (df.Projet=="RelOps")].Name.to_list())

df.groupby('Projet_Role').agg(mean_accuracy = ('Accuracy',np.mean), 
                              std_acccuracy = ('Accuracy', np.std), num_mach = ('Accuracy', len)).sort_values(by='mean_accuracy', ascending = False)


# # Find machines by Projet_Role and plot

# +
def alias_proj_role(proj_role):
    return df[df.Projet_Role == proj_role].Name.values

alias_enocienoci = alias_proj_role('ENO_CIENO_CI')
plot(alias_enocienoci,dataset,preds_df,len(test_all))
plot_activity_class(alias_enocienoci, gdth_df_cat, preds_df_cat) 
# -

alias_relopsodtreplay = alias_proj_role('RelOpsODT Replay')
plot(alias_relopsodtreplay,dataset,preds_df,len(test_all))
plot_activity_class(alias_relopsodtreplay, gdth_df_cat, preds_df_cat) 

alias_relopsodtreplay = alias_proj_role('ENOCIBuild')
plot(alias_relopsodtreplay,dataset,preds_df,len(test_all))
plot_activity_class(alias_relopsodtreplay, gdth_df_cat, preds_df_cat)



