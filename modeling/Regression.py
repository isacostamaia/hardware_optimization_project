import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO

import torch
from torch import nn, optim

from retrieve_and_prepare_data import db_to_df, interpolate_and_filter, split_data, prepare_data
from output import get_loss_plot, get_predict_plot
import settings



# NAME = str(settings.LR) + "_" + str(settings.EPOCHS) + "_" +

def get_batches(data, window):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window). 
    """

    L = len(data)
    for i in range(L - window):
        x_sequence = data[i:i + window]
        y_sequence = data[i+1: i + window + 1] 
        yield x_sequence, y_sequence

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hs):
   
        out, hs = self.lstm(x, hs)           # out.shape = (batch_size, seq_len, hidden_size)
        out = out.view(-1, self.hidden_size) # out.shape = (seq_len, hidden_size)     
        out = self.fc(out)
        
        return out, hs

def train(model, epochs, train_set, train_window, valid_data=None, lr=0.001, print_every=100):

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    valid_loss = []
    
    for e in range(epochs):
        
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
            
        if valid_data is not None:
                model.eval()
                val_x, val_y = valid_data
                val_x = val_x.unsqueeze(0)
                preds, _ = model(val_x, hs)
                v_loss = criterion(preds, val_y)
                valid_loss.append(v_loss.item())
                
                model.train()
            
        train_loss.append(np.mean(t_loss))
            
            
        if e % print_every == 0:
            print(f'Epoch {e}:\nTraining Loss: {train_loss[-1]}')
            if valid_data is not None:
                print(f'Validation Loss: {valid_loss[-1]}')

    return train_loss,valid_loss

    #get figure and save
    # fig64 = get_loss_plot(train_loss, valid_loss)
    # im = Image.open(BytesIO(base64.b64decode(fig64)))

    # #save figure
    # # results_directory = os.path.abspath(DIRECTORY)
    # results_directory = MACHINE_DIRECTORY
    # # # Create output directory if not existing
    # if not os.path.isdir(results_directory):
    #     os.makedirs(results_directory)

    # im.save('{}/loss.png'.format(results_directory), 'PNG')

def predict(model, train_set, test_set, train_window,scaler_test):

    fut_pred = len(test_set)

    test_inputs = (train_set[-train_window:].squeeze().tolist())

    model.eval()

    #  Get predictions on test data
    hs = None
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:]).unsqueeze(1)
        with torch.no_grad():
            test_preds, hs = model(seq.unsqueeze(0), hs)
            test_inputs.append(test_preds[-1].item()) #get last prediction

    preds = test_inputs[-len(test_set):]
    preds = scaler_test.inverse_transform(torch.FloatTensor(preds).reshape(-1, 1).detach()) 
    return preds

def prediction_one_machine(df):
    '''
        do prediction cycle from a dataframe of a single machine
    '''

    #interpolate and filter
    df_filt, df_non_filt = interpolate_and_filter(df)
    df_input = df_non_filt


    #DISONS QUE On va toujours calculer a partir des donnees non filtrees
    # if (settings.USE_FILTERED_DATA): 
    #     df_input = df_filt

    #split data
    train_df,test_df = split_data(df_input)

    #scale and reshape
    train_set,scaler_train = prepare_data(train_df)
    test_set, scaler_test =  prepare_data(test_df)

    #create model
    model = LSTM(settings.INPUT_SIZE, settings.HIDDEN_SIZE, settings.NUM_LAYERS, settings.OUTPUT_SIZE)

    #create validation set
    valid_data = test_set[-(settings.TRAIN_WINDOW+1):]
    valid_x = valid_data[:-1]
    valid_y = valid_data[1:]
    valid_data = (valid_x, valid_y)

    #train model
    train_loss,valid_loss = train(model,
                                settings.EPOCHS,
                                train_set,
                                settings.TRAIN_WINDOW,
                                valid_data=valid_data,
                                lr = settings.LR)

    #loss plot values                           
    loss_plot = [train_loss,valid_loss]

    #predict values
    preds = predict(model,train_set,test_set, settings.TRAIN_WINDOW, scaler_test)

    #predictions plot values
    truth_time = np.arange(0,len(df_input),1)
    y_truth_input = df_input.cpu
    y_truth_filtered = df_filt.cpu
    xy_truth_input = [truth_time,y_truth_input]
    
    test_time = np.arange(len(train_set)+1,len(train_set)+len(test_set)+1,1)
    y_test = preds.squeeze()
    xy_test = [test_time, y_test]

    pred_plot = [xy_truth_input, xy_test, y_truth_filtered]


    return loss_plot, pred_plot

    #get figure and save
    # fig64 = get_predict_plot(truth_time, y_truth, test_time, y_test)
    # im = Image.open(BytesIO(base64.b64decode(fig64)))
    # #save plot predictions
    # # results_directory = os.path.abspath(DIRECTORY)
    # results_directory = MACHINE_DIRECTORY 
    # # # Create output directory if not existing
    # if not os.path.isdir(results_directory):
    #     os.makedirs(results_directory)

    # im.save('{}/predictions_and_ground_truth_as_input.png'.format(results_directory), 'PNG')

    #plot predictions computed (if computed with raw) vs filtered ground truth
    # if(not settings.USE_FILTERED_DATA):
    #     y_truth_filtered = df_filt.cpu
    #     fig64 = get_predict_plot(truth_time, y_truth_filtered, test_time, y_test)
    #     im = Image.open(BytesIO(base64.b64decode(fig64)))

    #     im.save('{}/predictions_and_ground_truth_filtered.png'.format(results_directory), 'PNG')

# Run as program
if __name__ == '__main__':
    # for wip329dsy
    df = db_to_df()
    prediction_one_machine(df)



