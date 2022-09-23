from datetime import datetime, timedelta
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from data_source.dataframe import hosts_timeseries

def retrieve_df():
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
        'machines_to_include': None,
        # List of host names
        'machines_to_exclude': ['wip132dsy', 'ssdwip017dsy', 'ssdwip021dsy', 'ssdwip022dsy'],
        # Max number of records to fetch
        'limit': None
    }

    query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
    records = query.all()


    df_wks = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

    return df_wks

def df_to_treated_tensor(df_wks):
    #interpolate values and get same date intervals for all
    dfs, hostnames =  hosts_timeseries(df_wks)
    dfs = [a.rename(h) for a,h in zip(dfs,hostnames)]
    dataset = pd.concat(dfs,axis=1)

    # print("Dataset before scaling: \n", dataset.head(15))
    #scale values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_array_scaled = scaler.fit_transform(dataset)
    dataset_scaled = dataset.copy()
    dataset_scaled.loc[:,:] = dataset_array_scaled
    # print("Dataset after scaling: \n", dataset_scaled.head(15))

    tensor_dataset = torch.tensor(dataset_scaled.values)

    return tensor_dataset, hostnames


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
#         y = self.data[index+1: index + self.window + 1]
        y = self.data[index + self.window]
        return x , y
    
def get_batches_dataloader(data, window, batch_size):
    dataset = MyDataset(data.float(), q = window)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last=True)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

#         self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        #Embed input words
#         embedded = self.embedding(inputs)
        embedded = inputs
        #Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, device,  batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))

class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size+self.output_size, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        print("#####################ATTENTION##########################")
        encoder_outputs = encoder_outputs.squeeze()
        #Embed input words
        embedded = inputs.view(1, -1)
#         embedded = self.embedding(inputs).view(1, -1)
        embedded = self.dropout(embedded)
        print("             ##entries of tanh layer")
        print("encoder_outputs ",encoder_outputs)
        print("hidden[0] ",hidden[0])
        print("             ##end_entries of tanh layer")
        #Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0])+self.fc_encoder(encoder_outputs))

        print("             ##multiplication of result tanh layer and weights")
        alignment_scores = x.bmm(self.weight.unsqueeze(2))  
        print("alignment_scores ", alignment_scores)
        print("             ##end_multiplication of result tanh layer and weights")

        #Softmaxing alignment scores to get Attention weights
        print("             ##softmax layer")
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
        print("attn_weights ", attn_weights)
        print("             ##end_softmax layer")
        

        #Multiplying the Attention weights with encoder outputs to get the context vector
        print("             ##context vector: mult attn-weights and encoder_outputs")
        context_vector = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        print("context_ vector ",context_vector)
        print("             ##end_context vector: mult attn-weights and encoder_outputs")

        #Concatenating context vector with embedded input word
        print("             ##concatenation of dec input (previous dec output) and context vector")
        output = torch.cat((embedded, context_vector[0]), 1).unsqueeze(0)
        print("output",output)
        print("             ##end_concatenation of dec input (previous dec output) and context vector")

        #Passing the concatenated vector as input to the LSTM cell
        print("             ##output of lstm layer")
        output, hidden = self.lstm(output, hidden)
        print("output[0] ",output[0])
        print("             ##end_output of lstm layer")
        #Passing the LSTM output through a Linear layer acting as a classifier
        print("             ##result of last linear layer")
        output = self.classifier(output[0])
        print("output", output)
        print("             ##end_result of last linear layer")

        print("#####################endATTENTION##########################")
        return output, hidden, attn_weights

class LuongDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=1, drop_prob=0.1):
        super(LuongDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        #Our Attention Mechanism is defined in a separate class
        self.attention = attention

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size*2, self.output_size)
    
    def forward(self, inputs, hidden, encoder_outputs):
        #Embed input words
        embedded = self.embedding(inputs).view(1,1,-1)
        embedded = self.dropout(embedded)

        #Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        #Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(hidden[0], encoder_outputs)
        #Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)

        #Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs)

        #Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector),-1)
        #Pass concatenated vector through Linear layer acting as a Classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attn_weights

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        #Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
  
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
          #For the dot scoring method, no weights or linear layers are involved
          return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
    
        elif self.method == "general":
            #For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)

        elif self.method == "concat":
            #For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


def train(encoder, 
        decoder, 
        encoder_optimizer, 
        decoder_optimizer, 
        train_all, 
        train_window, 
        device, 
        hostnames):

    EPOCHS = 10
    teacher_forcing_prob = 0.5
    encoder.train()
    decoder.train()

    data = get_batches_dataloader(train_all, train_window, batch_size = 1)
    criterion = nn.MSELoss()

    avg_loss = []
    for epoch in range(EPOCHS):
        print("*********************************EPOCH ", epoch)
        samples_errors = []
        for  x, y in data:
            print("x :",x)
            loss = 0.
            h = encoder.init_hidden(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            if (len(x.shape)==2):
                x = x.unsqueeze(0)
            inp = x.to(device)

            print("x shape",inp.shape)
            encoder_outputs, h = encoder(inp,h)
            print("encoder_outputs ",encoder_outputs)
            print("encoder_outputs shape",encoder_outputs.shape)
            
            #First decoder input will be the SOS token
            decoder_input = torch.zeros(1 , len(hostnames), 1, device=device)
            #First decoder hidden state will be last encoder hidden state
            decoder_hidden = h
            output = []
            teacher_forcing = True if random.random() < teacher_forcing_prob else False
            
    #         for ii in range(1):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            print("decoder_output ", decoder_output)
            print("decoder_output shape",decoder_output.shape)
            print("y ", y)
            print("y shape ", y.shape)
    #             Get the index value of the word with the highest score from the decoder output
            top_value, top_index = decoder_output.topk(1)
            print("top_value ", top_value)
            print("top_index ", top_index)
    #             if teacher_forcing:
    #                 decoder_input = torch.tensor([de_inputs[i][ii]],device=device)
    #             else:
            # decoder_input = torch.tensor([top_index.item()],device=device)
            decoder_input = decoder_output
            print("decoder_input ",decoder_input)
            print("decoder_input shape",decoder_input.shape)
            output.append(top_index.item())
            #Calculate the loss of the prediction against the actual word
            loss = criterion(decoder_output.view(1,-1), y.to(device))
            print("LOSS ",loss)
            print(loss)
            loss.backward()
            samples_errors.append(loss.item())
            print(samples_errors)
            encoder_optimizer.step()
            decoder_optimizer.step()
        avg_loss.append(np.sum(samples_errors)/data.__len__())

    return avg_loss


def do_all():
    #prepare device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    #prepare dataset
    df_wks = retrieve_df()
    tensor_dataset, hostnames = df_to_treated_tensor(df_wks)
    train_all, test_all, valid_all = split_data(tensor_dataset)

    #Define network
    hidden_size = 500
    encoder = EncoderLSTM(len(hostnames), hidden_size).to(device)
    decoder = BahdanauDecoder(hidden_size,len(hostnames)).to(device)
    lr = 0.00025
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    train_window = 200
    avg_loss = train(encoder, 
                decoder, 
                encoder_optimizer, 
                decoder_optimizer, 
                train_all, 
                train_window, 
                device, 
                hostnames)
    print("\n############################\n AVG LOSS \n", avg_loss)
    plt.plot(avg_loss)
    plt.savefig("loss_att.png")




# Run as program
if __name__ == '__main__':
    do_all()