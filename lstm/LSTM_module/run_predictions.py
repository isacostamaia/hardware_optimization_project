
import numpy as np
import pandas as pd
import torch

from .run_train import run_train
from . import settings




def get_scaled_predictions(model, dm):
    '''
        Retrieve untreated predictions
    '''
    # Get cpu or gpu device for training.
    device = "cuda" if (torch.cuda.is_available() and settings.GPU_DESIRED) else "cpu"
    print("Using {} device".format(device))

    prediction_range = 3

    train_window = settings.PARAM['seq_len']
    fut_pred = len(dm.test_all)

    test_inputs = (dm.train_all[-train_window:])

    #size of a single prediction
    preds = torch.tensor([])
    all_preds = torch.tensor([])

    if device == 'cuda':
        model.to(device)
        test_inputs = test_inputs.cuda().to(device)
        preds = preds.cuda().to(device)
        all_preds = all_preds.cuda().to(device)
        test_all = dm.test_all.cuda().to(device)

    model.eval()

    #clear train hidden state for compatibility in case batch_size != 1
    model.previous_hidden = None

    #  Get predictions on test data
    for i in range(fut_pred):
        

        #get last predictions by getting values in preds tensor and the number of ground truth values needed to complete train_window
        #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
        seq = torch.cat((test_inputs[-(train_window - preds.shape[0]):], preds),0).unsqueeze(0).float()
        
        if device == 'cuda':
            seq = seq.cuda().to(device)
        
        with torch.no_grad():
            test_preds = model(seq)
            #if it is three dimensional, take out the first dimension (that will be equal 1, which is the batch_size)
            test_preds = test_preds.squeeze()

            #get last value (prediction) and put in preds vector
            preds = torch.cat((preds, test_preds[-1].unsqueeze(0)),0)
            #concatenate in all predictions tensor
            all_preds = torch.cat((all_preds, test_preds[-1].unsqueeze(0)),0)

            if device == 'cuda':
                preds = preds.cuda().to(device)

            #preds: tensor with predictions used to make more predictions. Its first dimension len gets till train_window
            #test_inputs: tensor with ground truth values used to make predictions
            if (preds.shape[0]==prediction_range):
                preds = torch.tensor([])
                if device == 'cuda':
                    preds = preds.cuda().to(device)
                test_inputs = torch.cat((test_inputs,test_all[i:i+prediction_range]),0)


    return all_preds

def treat_predictions(dm, tests_fin):
    
    #transform predictions tensor to dataframe
    # tests_fin.shape # len(Test_all), num_machines
    preds_df = pd.DataFrame(tests_fin.cpu().numpy())
    
    #undo predictions rescaling
    preds_array_unscaled = dm.scaler.inverse_transform(preds_df)
    dic = {i:h for i,h in zip(np.arange(0,preds_df.columns.stop), dm.hostnames)}
    preds_df = preds_df.rename(columns=dic)
    
    #replace predictions unscaled values in dataframe
    preds_df= pd.DataFrame(preds_array_unscaled, columns= dm.hostnames)
    
    return preds_df

def get_predictions(model, dm):
    '''
        Return prediction dataframe
    '''

    return treat_predictions(dm, get_scaled_predictions(model, dm))

# Run as program
if __name__ == '__main__':
    model, dm = run_train()
    preds_df = get_predictions(model, dm)
    