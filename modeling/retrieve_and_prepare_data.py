from datetime import datetime, timedelta
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import preprocessing
import torch
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df, hosts_freqseries



def db_to_df():

    # list_mach = ['wip196dsy','ssdhwip017dsy', 'pool27dsy']#,'wip212dsy','wip214dsy',,'pool36dsy']

    # Set interval & filters
    query_params = {
        # 'day', 'hour', 'minute'
        'interval': 'hour',
        # datetime
        'start_date': datetime.now() - timedelta(weeks = 10),
        # datetime
        'end_date': None,
        # 'windows', 'linux'
        'os': 'windows',
        # List of host names
        'machines_to_include': None, #['wip329dsy'],
        # List of host names
        'machines_to_exclude': None,
        # Max number of records to fetch
        'limit': 100
    }

    query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
    records = query.all()
    df = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

    #get further info from alias
    machine_suivi_path = 'resources/Machines_Suivi.csv' 
    machines_df = pd.read_csv(machine_suivi_path, sep=';')
    machines_df.loc[:, 'Name'] = [x.lower() for x in machines_df.Name]
    machines_df = machines_df[['Name', 'Projet', 'Role']]
    df = df.copy().merge(
        machines_df,
        how='left',
        left_on='hostname',
        right_on='Name'
    )

    return df

def interpolate_and_filter(df):
    '''
        interpolates and filter signal from a given dataframe
    '''
    
    #if we want to use a interpolated signal
    df_interp = hosts_freqseries(df)[0]
    interpolated_signal = df_interp.cpu.values

    #if we want to use the interpolated signal filtered

    interpolated_signal_filtered = savgol_filter(interpolated_signal, 21, 3) # window size 51, polynomial order 3
    df_interp_filt = pd.DataFrame({'date': df_interp.index, 'cpu': interpolated_signal_filtered}).set_index('date')
    
    return df_interp_filt , df_interp


def split_data(df):
    '''
        returns train, test dataframes.
        Obs:  (index where test starts is clearly len(train)+1)
    '''
    per = 0.05
    len_train = int((1-per)*len(df))
    train = df.iloc[:len_train]
    test = df.iloc[len_train:]
    return train, test


def prepare_data(df):
    '''
        returns scaled tensor
    '''
    data = df.cpu.to_numpy().reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    # convert data to tensor
    data = torch.tensor(data, dtype=torch.float32)
    return data, scaler