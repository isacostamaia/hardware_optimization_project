import io
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib.cm as cm
import scipy
from tslearn.metrics import dtw
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error

# from LSTM_module import settings

VERSION_NAME = 'attention_trash'
GLOBAL_DIREC = './lstm/'

def error_peaks(gd_th, pred):
    '''
        takes into account only errors from peaks, weighted by peak value importance wrt the gd th mean and do the sum
    '''
    error = []
    
    #peaks
    peaks, _ = find_peaks(gd_th, prominence=12)
    gd_th_peaks = gd_th[peaks]
    pred_peaks = pred[peaks]
    gd_th_mean = np.mean(gd_th)
    for g, p in zip(gd_th_peaks,pred_peaks):
            error.append((np.abs(g-p)/g)*(g/gd_th_mean))
            
    if len(error)==0:
        return 0
    
    return np.sum(np.array(error))

def compute_test_error(dataset,preds_df,len_train):
    '''
        compute dtw distance and mixed error between ground truth and predicted 
        values for the test set. 
        Return value is a tuple where each position is a list of errors for all the machines.
    '''


    dist_dtw_ = []
    mixed_err_ = []
    for hostname in dataset.columns:
        #ground truth
        gd_th_mach_test = dataset.loc[:, hostname].copy()
            #test slice 
        gd_th_mach_test = gd_th_mach_test.iloc[len_train:len_train+len(preds_df)]
        gd_th_mach_test = gd_th_mach_test.values

        #predictions
        pred = preds_df.loc[:,hostname].to_numpy()

        #for dtw distance
        distance_dtw = dtw(gd_th_mach_test, pred)
        
        #for mse
        mse = mean_squared_error(gd_th_mach_test, pred)
        

        
        dist_dtw_.append(distance_dtw)
        mixed_err_.append(mse)

    return dist_dtw_, mixed_err_

rcParams.update({'figure.autolayout': True})

_DPI = 96
_figure, _axix = plt.subplots()

def get_machine_figure(hostname,dataset,preds_df,len_train):

    #ground truth
    gd_th_mach = dataset.loc[:, hostname].copy()
    gd_th_mach = gd_th_mach.iloc[:len_train+len(preds_df)]
    gd_th_mach = gd_th_mach.values
    x_gd_th = np.arange(0,len_train+len(preds_df),1)
    
    #predictions
    pred = preds_df.loc[:,hostname]
    x_pred = np.arange(len_train, len_train+len(pred),1)

    #plot
    _axix.plot(x_gd_th,gd_th_mach,  label='Ground Truth' )
    _axix.plot(x_pred, pred, 'r--', label='Test Predictions')

    #to compute peaks
    df_mach_test = (gd_th_mach[len_train:]) #df_mach goes till end of test set
    df_mach_test = df_mach_test
    pred = pred.to_numpy()

    #peaks
    peaks, _ = find_peaks(df_mach_test, prominence=12, distance=1)
    _axix.plot(peaks+len_train, df_mach_test[peaks], "x", label = 'peaks') #, linewidth=0.5

    # leg = plt.legend()

    _axix.set_ylim([-10,75])

    _figure.set_size_inches(1800/_DPI, 210/_DPI) #600

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='png')
    plt.cla()
    io_bytes.seek(0)
    base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
    io_bytes.close()
    return base64_image


def generate_predictions_html(error, dataset, preds_df, len_train):
    """
    Generate an html file for a model
    error: tuple containing dist_dtw_, mixed_err_ lists following the dataframe.columns order
    dataset: dataframe whose columns are machines and rows are dateindex
    preds_df: prediction dataframe
    len_train: len of train set

    """
    # global_direc = './lstm/'
    direc = GLOBAL_DIREC + '{}/'.format(VERSION_NAME)

    html = '<div style="display: flex;flex-wrap: wrap;font-family: calibri">'

    for h, dtw, mix in zip(dataset.columns, error[0], error[1]):
        figure_b64 = get_machine_figure(h, dataset, preds_df, len_train)
        html += (
            '<div>' +
                '<div>' +
                    '{} - dtw = {} | mse = {}'.format(h, round(dtw,2), round(mix,2)) +
                '</div>' +
                '<img src="data:image/png;base64,{0}">'.format(figure_b64) +
            '</div>'
        )

    html += '</div>'

    file_name = direc + 'predictions.html'

    with open(file_name, 'w', errors='replace') as html_file:
        html_file.write(html)


# def get_loss_img(): 
#     _figure, _axix = plt.subplots(figsize=(16, 5))

#     metrics = pd.read_csv('./lstm/{}/metrics.csv'.format(version_name))
#     train_loss = metrics[['Train_loss_epoch', 'step', 'epoch']][~np.isnan(metrics['Train_loss_epoch'])]
#     val_loss = metrics[['Validation_loss_epoch', 'epoch']][~np.isnan(metrics['Validation_loss_epoch'])]

#     _axix.set_title('Loss per epoch')
#     _axix.plot(train_loss['epoch'], train_loss['Train_loss_epoch'], label = "Train")
#     _axix.plot(val_loss['epoch'], val_loss['Validation_loss_epoch'], color='orange', label = "Validation")

#     _axix.legend()
#     _figure.set_size_inches(1200/_DPI, 400/_DPI)

#     io_bytes = io.BytesIO()
#     plt.savefig(io_bytes, format='png')
#     plt.cla()
#     io_bytes.seek(0)
#     base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
#     io_bytes.close()
#     return base64_image

def get_loss_img(avg_loss):
    _axix.set_title('Loss per epoch')
    _axix.plot(avg_loss, label = "Train")

    _axix.legend()
    _figure.set_size_inches(1200/_DPI, 400/_DPI)

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='png')
    plt.cla()
    io_bytes.seek(0)
    base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
    io_bytes.close()
    return base64_image
