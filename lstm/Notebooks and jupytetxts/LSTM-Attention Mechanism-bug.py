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
import torch.nn.functional as F
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
##################################
#we'll try to reproduce 2051 model
##################################

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

##################################
#we'll try to reproduce 7395 model
##################################

# Set interval & filters
query_params = {
    # 'day', 'hour', 'minute'
    'interval': 'hour',
    # datetime
#     'start_date': datetime.now() - timedelta(weeks =9),
    'start_date': datetime(2021, 3, 25, 11, 0) , 
    # datetime
    'end_date': datetime(2021, 3, 25, 11, 0) + timedelta(weeks = 9),
    # 'windows', 'linux'
    'os': 'windows',
    # List of host names
    'machines_to_include':machines_model,
    # List of host names
    'machines_to_exclude': ['wip132dsy', 'ssdwip017dsy', 'ssdwip021dsy', 'ssdwip022dsy'],
    # Max number of records to fetch
    'limit': None
}

query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
records = query.all()
# -

df_wks = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

len(df_wks.hostname.unique())

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
# dataset.loc[:, 'client14dsy'].values

# +
# def get_batches(data, window):
#     """
#     Takes data with shape (n_samples, n_features) and creates mini-batches
#     with shape (1, window). 
#     """

#     L = len(data)
#     for i in range(L - window):
#         x_sequence = data[i:i + window]
#         y_sequence = data[i+1: i + window + 1] 
#         yield x_sequence, y_sequence;
        
class MyDataset(Dataset):
    def __init__(self, data, q):
        self.data = torch.Tensor(data)
        self.window = q

    def __len__(self):
        return len(self.data) -  self.window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
#         y = self.data[index+1: index + self.window + 1]
        y = self.data[index + self.window]
        return x , y
    
def get_batches_dataloader(data, window, batch_size):
    dataset = MyDataset(data.float(), q = window)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last=True)

train_dataset = MyDataset(train_all.float(),q=5)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)

# valid_dataset = MyDataset(valid_all.float(),q=train_window)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)

# train_loader = get_batches_dataloader(train_all, train_window, batch_size = 1)
# valid_loader = get_batches_dataloader(valid_all, train_window, batch_size = 1)

# for x, y in (train_loader):
#     print(x, y)
#     print(x.shape, y.shape)
#     break

# +
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.num_layers = num_layers
        
    def forward(self, x, hs):
        print("#############LSTM################")
        print("x input LSTM shape: ", x.shape)
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
#         out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size)     

        print("out after LSTM shape ", out.shape)
        print("##########endLSTM################")

        return out, hs
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers , 1, self.hidden_size),
                torch.zeros(self.num_layers , 1, self.hidden_size))
    

class BahdanauDecoder(nn.Module): #obs: output_size is output_seq_len.NO.I change for the number of machines.
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.input_size = input_size #number of machines 
        #I use this above to conactenate all the machines output with the context vector, since we only 
        #have one context vector for all of them

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size + self.input_size, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        #inputs are actually the previous decoder outputs
        
        print("#############ATTE################")
        encoder_outputs = encoder_outputs.squeeze()
        
        #no longer that:
        # Embed input words
#         embedded = self.embedding(inputs).view(1, -1)
        #instead:
        embedded = inputs[0].clone() #shape (#num_machines, 1)
        
        embedded = self.dropout(embedded)

        # Calculating Alignment Scores
        hidden = tuple(h.detach() for h in hidden)
        x = torch.tanh(self.fc_hidden(hidden[0])+self.fc_encoder(encoder_outputs))
        alignment_scores = x.bmm(self.weight.unsqueeze(2))  

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
        
        print("encoder_outputs shape ", encoder_outputs.shape)
        print("attn_weights shape ", attn_weights.shape)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Concatenating context vector with embedded input word
        print("embedded (inputs[0]) shape ", embedded.shape)
        print("context vector shape ", context_vector.shape)
        #we only have one context vector for all the machines at all time steps in the input sequence
        print("dont forget im using context_vector[0] in cat")
        
        
        #no longer that:
#         output = torch.cat((embedded, context_vector[0]), 1).unsqueeze(0)
        #instead:
#         lines = []
#         for k in range(embedded.shape[0]):
#             l = torch.cat((embedded[k].clone().unsqueeze(0), context_vector[0].clone()), dim = 1).detach()
#             lines.append(l.detach())

#         output = ((torch.cat(lines)).unsqueeze(0)).detach()
        
        output = torch.cat((embedded.view(1,-1), context_vector[0]), 1).unsqueeze(0)
        
        # Passing the concatenated vector as input to the LSTM cell
        print("decoder lstm input (i.e. concat of decoder output and context vector) shape ", output.shape)
        
        output, hidden = self.lstm(output, hidden)
        print("decoder lstm output shape ", output.shape)
        
        #no longer that:
        # Passing the LSTM output through a Linear layer acting as a classifier
#         output = F.log_softmax(self.classifier(output[0]), dim=1)
        #instead:
        output = self.classifier(output[0]) #.clone().detach()
        print("decoder linear output shape ", output.shape)
        
        print("##########endATTE################")
        return output, hidden, attn_weights
    
    def init_in(self):
        return torch.zeros(1 , input_size, 1)
    
    def init_hidden(self):
        return (torch.zeros(1 , 1, hidden_size),  
                         torch.zeros(1 , 1, hidden_size)) 
# +
# # ###############################################
# # # TOY CASE
# # ##############################################


# input_size = len(hostnames)
# hidden_size = 500
# num_layers = 4
# # output_seq_len = 1
# output_seq_len = len(hostnames)
# train_window = 32 #equiv to vocab_size

# for s, l in get_batches_dataloader(train_all, train_window, batch_size = 1):
#     break
# s.shape


# c = LSTM(input_size, hidden_size)
# a, b = c.forward(s, c.init_hidden()) #
# print(a.shape)
# print(b[0].shape)
# print(b[1].shape)

# out = torch.zeros(1 , input_size, 1) # has size of out 
# #(out of decoder is embbeded in hidden_size shape so it can be concatenated with context vector)
# #this is the previous decoder output that serves as "input" in the forward

# hidden = (torch.zeros(1 , 1, hidden_size),  # hidden state is only the last one of the sequence
#      torch.zeros(1 , 1, hidden_size))       # so even though we have 32 hidden_states (in "out"), the hidden is only the last one

# att = BahdanauDecoder(hidden_size, output_seq_len)
# o, h, aw = att.forward(out, hidden, a)
# print(o.shape)
# print(h[0].shape)
# print(h[1].shape)
# print(aw.shape)

# +
def train(model_encoder, model_decoder, epochs, 
          train_set, train_window, device, 
          valid_data=None, lr=0.001, batch_size =1, 
          print_every=10, loss_fct = 'MSE'):

    if device == 'cuda':
        model_encoder.cuda()
        model_decoder.cuda()
        
    
    if loss_fct == "MSE":
        criterion = nn.MSELoss()
    elif loss_fct == "MAPE":
        criterion = MAPE()
    elif loss_fct == "DTW":
        criterion = SoftDTW(gamma=0.0001, normalize=False, use_cuda = True) # just like nn.MSELoss()

    
#appereantly MAPE cannot deal with 3d data to compute loss...do an average with each one of the samples?


#     torch.autograd.set_detect_anomaly(True)
    opt_encod = optim.Adam(model_encoder.parameters(), lr=lr)
    opt_decod  = optim.Adam(model_decoder.parameters(), lr=lr)
    
    train_loss = []
    valid_loss = []
    
    with torch.autograd.set_detect_anomaly(True):  
        for e in range(epochs):
            print("EPOCH {} ****************************\n".format(e))

            #initial values
            hs_encoder = model_encoder.init_hidden()
            in_decoder = model_decoder.init_in()
#             hs_decoder = model_decoder.init_hidden()

            if device == 'cuda':
                hs_encoder = tuple([i.cuda().to(device) for i in hs_encoder]) 
                in_decoder = in_decoder.cuda().to(device)
    #             hs_decoder = tuple([i.cuda().to(device) for i in hs_decoder])

            t_loss = []
            for x, y in get_batches_dataloader(train_set, train_window, batch_size = batch_size):

                if device == 'cuda':
                    x = x.cuda().to(device)
                    y = y.cuda().to(device)



                # Create batch_size dimension if it doesn't exists
                if (len(x.shape)==2):
                    x = x.unsqueeze(0)

                out_encoder, hs_encoder = model_encoder(x, hs_encoder)
                hs_decoder = hs_encoder
                out_decoder, hs_decoder, aw = model_decoder(in_decoder, hs_decoder, out_encoder)
                out_decoder = out_decoder.unsqueeze(0).detach().clone()
                in_decoder = out_decoder.clone()

    #             hs = tuple([h.data for h in hs])


                print("out_decoder shape", out_decoder.shape)

                print("y shape ", y.shape)
                print("out_decoder.view(1,1,-1) shape", out_decoder.shape)

                loss = criterion(out_decoder, y.unsqueeze(0).view(1,1,-1))
                loss.requires_grad = True
                print(loss)
                if loss_fct == 'DTW':
                    loss = torch.mean(loss)
                    print("mean batch loss",loss)

                opt_encod.zero_grad()
                opt_decod.zero_grad()
                print("um")
                #loss.backward(retain_graph=True)
                loss.backward()
                print("dois")
                opt_encod.step()
                print("tres")
                opt_decod.step()
                print("quatro")
                t_loss.append(loss.item())
                print("cinco")

#         v_loss = []
#         if valid_data is not None:
#             print(" \n Entered in validation \n")
#             model_encoder.eval()
#             model_decoder.eval()
            
#             #initial values
#             hs_encoder_val = model_encoder.init_hidden()
#             out_decoder_val = model_decoder.init_out()
#             hs_decoder_val = model_decoder.init_hidden()
            
#             if device == 'cuda':
#                 hs_encoder_val = [i.cuda().to(device) for i in hs_encoder_val] 
#                 out_decoder = out_decoder_val.cuda().to(device)
#                 hs_decoder_val = [i.cuda().to(device) for i in hs_decoder_val]
            
#             for val_x, val_y in get_batches_dataloader(valid_data, train_window, batch_size =batch_size):
#                 if device == 'cuda':
#                     val_x = val_x.cuda().to(device)
#                     val_y = val_y.cuda().to(device)
                
#                 with torch.no_grad():
#                     if (len(val_x.shape)==2):
#                         val_x = val_x.unsqueeze(0)
                        
                        
#                     out_encoder_val, hs_encoder_val = model_encoder(val_x, hs_encoder_val)
#                     out_decoder_val, hs_decoder_val, aw = model_decoder(out_decoder_val, hs_decoder_val, out_encoder_val)
#                     out_decoder_val = out_decoder_val.unsqueeze(0)
                    
# #                     preds, hs_val = model(val_x, hs_val)
# #                     hs_val = tuple([h.data for h in hs_val])
                    
#                     #to compute mape loss
#                     if loss_fct == "MAPE":
#                         loss_val = torch.zeros(1)
#                         if device == 'cuda':
#                             loss_val = loss_val.cuda().to(device) 
#                         for b in range(batch_size):
                            
#                             preds_one_of_batch = preds[b,:,:]
#                             val_y_one_of_batch = val_y[b,:,:]
#                             loss_val  = loss_val + criterion(preds_one_of_batch, val_y_one_of_batch)
#                         loss_val = loss_val/batch_size

                        
#                     else:
#                         loss_val = criterion(out_decoder_val.view(1,1,-1), val_y.unsqueeze(0))
#                         print(loss)
#                         if loss_fct == 'DTW':
#                             loss = torch.mean(loss)
#                             print("mean batch loss",loss)
                        
#                     v_loss.append(loss_val.item())
            
                

#             valid_loss.append(np.mean(v_loss)) #mean of scalar is the scalar itself
            

#             model_encoder.train()
#             model_decoder.train()
            
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

input_size = len(hostnames)
hidden_size = 500
# output_seq_len = 1
output_seq_len = len(hostnames)
train_window = 4
# num_layers = 2
# output_size = len(hostnames)


################defining model
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


model_encoder = LSTM(input_size, hidden_size)
model_encoder = model_encoder.float()

model_decoder = BahdanauDecoder(hidden_size, output_seq_len)
model_decoder = model_decoder.float()
if device == 'cuda':
    model_encoder = model_encoder.to(device) 
    model_decoder = model_decoder.to(device) 
#############################


#using data with batch size already included
epochs = 250
learning_rate= 0.00025
# train(model, epochs, train_all, train_window, device, valid_data=valid_all ,lr=0.0005, batch_size = 1, print_every=1)

train(model_encoder, model_decoder, epochs, 
          train_all, train_window, device, 
          valid_data=valid_all, lr=learning_rate, batch_size =1, 
          print_every=10, loss_fct = 'MSE')

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

# ##  evaluation by role

# +
import chardet

with open('Machines_Suivi.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


# pd.read_csv('filename.csv', encoding=result['encoding'])

machines_df = pd.read_csv('Machines_Suivi.csv', sep=';',  encoding=result['encoding'])

machines_df.loc[:, 'Name'] = [x.lower() for x in machines_df.Name]
machines_df = machines_df[['Name', 'Projet', 'Role']]

machines_df.head()
# -

dataset_test = dataset.iloc[len(train_all):len(train_all)+len(test_all)]
dataset_test.head()

preds_df.head()

# +
loss = nn.MSELoss()

mse_alias = [loss(
    torch.Tensor(preds_df.loc[:,alias].values), 
    torch.Tensor(dataset_test.loc[:,alias].values)
     ) 
 for alias in dataset.columns]

mse_alias = np.array(mse_alias)

results_df = pd.DataFrame({"Name": dataset.columns, "MSE": mse_alias})
results_df
# -

df = results_df.copy().merge(
    machines_df,
    how='left',
    on='Name'
)
df['Projet_Role'] =  df['Projet'] + df["Role"]
df.head()

df.groupby('Projet_Role').agg(mean_MSE = ('MSE',np.mean), std_MSE = ('MSE', np.std),
                              num_machines = ('MSE', len)).sort_values(by='mean_MSE')

enoci_TBD_alias = df[df.Projet_Role == 'ENO_CITBD'].Name.values

plot(enoci_TBD_alias[:10],dataset,preds_df,len(train_all))

plot(enoci_TBD_alias[10:20],dataset,preds_df,len(train_all))

monitor_slave_alias = df[df.Projet_Role == 'MoniteurMoniteur - slave'].Name.values
plot(monitor_slave_alias[:10],dataset,preds_df,len(train_all))







# +
def boxplot_sorted(df, by, column):
  df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
  meds = df2.mean().sort_values()
  df2[meds.index].boxplot(rot=90)

boxplot_sorted(df, by=['Projet_Role'], column="MSE")
# -

# ## data distribution (as from nagios)

# +
freq, bins, patches = plt.hist(dataset.values.ravel())

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

plt.xlabel("% CPU")
plt.title("% CPU distribution")
plt.show;
# -
# ## clustering results


preds_df

gdth_test_df = dataset.iloc[len(train_all):len(train_all) + len(test_all), :]
gdth_test_df

#categorize predictions
preds_res = preds_df.values.reshape(-1,1)
est_preds = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est_preds.fit(preds_res)
preds_res_cat = est_preds.transform(preds_res)
preds_res_cat
preds_cat = preds_res_cat.reshape(-1, preds_df.values.shape[1])
preds_df_cat  =  pd.DataFrame(preds_cat, columns = preds_df.columns)

print(preds_df_cat.apply(lambda col: col.unique()))

#categorize ground truth in test
gdth_res = gdth_test_df.values.reshape(-1,1)
est_gdth = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
est_gdth.fit(gdth_res)
gdth_res_cat = est_gdth.transform(gdth_res)
gdth_res_cat
gdth_cat = gdth_res_cat.reshape(-1, gdth_test_df.values.shape[1])
gdth_df_cat  =  pd.DataFrame(gdth_cat, columns = gdth_test_df.columns)

# +
#confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(gdth_df_cat.values.ravel(), preds_df_cat.values.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=[0,1,2])
disp.plot()

# +
all_alias = dataset.columns

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
        ax.set_ylim([0,3.5])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.9)

plot_activity_class(all_alias[:int(len(all_alias)/30)], gdth_df_cat, preds_df_cat)   
# -


