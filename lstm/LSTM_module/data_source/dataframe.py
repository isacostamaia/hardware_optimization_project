"""
Get formatted Nagios CPU time series dataframes
"""
import pandas as pd
import numpy as np

# from ..settings import (
    # DB_QUERY_ARGS
# )

FREQ_SAMPLING = dict(
    day  =    'D' ,
    hour =    'H',
    minute =  '5T'
    )





def db_to_df(records):
    """
    Convert SQLA DB records to a dataframe
    records: list or records of the form
    (date_1, host_name_1, os, average_cpu)
    Return a dataframe of the form
    (date_1, host_name_1, os, average_cpu)
    """
    return pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])

def interval_cpu_mean_df(df, interval_seconds):
    """
    Get a time series of CPU averages on the specified time interval

    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    interval_seconds: time series interval in seconds

    Return a dataframe of format:
    (date, cpu)
    """
    df_copy = df.copy()
    df_copy.date = pd.to_datetime(df_copy.date)
    df_date_cpu = df_copy[['date', 'cpu']].groupby(
        pd.Grouper(
            key='date',
            freq='{}S'.format(interval_seconds)
        )
    )
    df_date_cpu = df_date_cpu.mean()
    # Reinitialize index
    df_date_cpu = df_date_cpu.rename_axis('date').reset_index()
    return df_date_cpu

def hosts_timeseries(df, interval_seconds=None):
    """
    Get a list of cpu value timeseries for each host that exists in
    input dataframe
    For each host dataframe, fill missing values with adjacent ones
    df: dataframe of the form
    ('date1', 'hostname1', 'os', 'cpu')
    ('date1', 'hostname2', 'os', 'cpu')
    ...
    ('date2', 'hostname1', 'os', 'cpu')
    ('date2', 'hostname2', 'os', 'cpu')
    interval_seconds: if specified, return time series averaged on the specified interval
    Return a list of time series having the form
    [
        [
            cpu, # For date1 hostname1
            cpu, # For date2 hostname1
        ],
        [
            cpu, # For date1 hostname2
            cpu, # For date2 hostname2
        ],
        ...
    ]
    """
    dfs = []
    hostnames = df.hostname.unique()
    dates = df.date.unique()
    interval = pd.Timedelta(dates[1] - dates[0]).total_seconds()
    date_range = pd.date_range(dates[0], dates[-1], freq='{}S'.format(interval))
    for hostname in hostnames:
        host_df = df.loc[df.hostname == hostname].copy()
        host_df.date = pd.to_datetime(host_df.date)
        # Set date as the index
        host_df.set_index('date', inplace=True)
        # Fill missing dates
        host_df = host_df.reindex(date_range)
        # Set values of missing dates by interpolation
        # Pad method will use adjacent string/numeric values from the previous
        # adjacent index
        host_df.interpolate(method='pad', inplace=True)
        if interval_seconds:
            # Set back index to default
            host_df = host_df.rename_axis('date').reset_index()
            host_df = interval_cpu_mean_df(host_df, interval_seconds)
        host_df = host_df['cpu']
        # Fillna in case first elements are None
        host_df.fillna(0, inplace=True)
        dfs.append(host_df)
    return dfs, hostnames

def normalized_cpu_threshold_df_deprec(df, value_thresholds):
    """
    Get a time series of CPU values normalized against the specified
    value thresholds
    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    value_thresholds: array of ordered CPU threshold values ; eg. [5, 30, 60, 100]
    Return a dataframe of format:
    ('date', 'hostname', 'os', 'cpu')
    Ex. of ouput CPU values for value_thresholds = [5, 30, 60, 100]
    input: 2, output 0
    input: 4, output 0
    input: 5, output 0
    input: 12, output 1
    input: 55, output 2
    input: 62, output 3
    """
    df_copy = df.copy()
    minimum = 0

    for index, threshold in enumerate(value_thresholds):        
        df_copy.loc[(df_copy.cpu > minimum) & (df_copy.cpu <= threshold), 'cpu'] = index
        minimum = threshold
    return df_copy

def normalized_cpu_threshold_df(df, value_thresholds):
    """
    Get a time series of CPU values normalized against the specified
    value thresholds
    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    value_thresholds: array of ordered CPU threshold values ; eg. [5, 30, 60, 100]
    Return a dataframe of format:
    ('date', 'hostname', 'os', 'cpu')
    Ex. of ouput CPU values for value_thresholds = [5, 30, 60, 100]
    input: 2, output 0
    input: 4, output 0
    input: 5, output 25
    input: 12, output 25
    input: 55, output 50
    input: 62, output 75
    """
    df_copy = df.copy()
    minimum = 0

    increment = 100/len(value_thresholds)

    for index, threshold in enumerate(value_thresholds):        
        df_copy.loc[(df_copy.cpu > minimum) & (df_copy.cpu <= threshold), 'cpu'] = index*increment
        minimum = threshold
    return df_copy

def auto_interval_cpu_mean_df(df, max_rows):
    """
    Get a time series of CPU averages with interval automatically set to
    fit the specified max number of rows

    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    max_rows: maximum number of rows in output dataframe

    Return a (output_df, interval_seconds, interval_str) tuple
    output_df: (date, average_cpu) dataframe
    interval_seconds: interval in seconds
    interval_str: interval in the form '<n>day, <n>hour, <n>minute'
    """
    time_span_seconds = (df.date.iloc[-1] - df.date.iloc[0]).total_seconds()
    interval_seconds = int(time_span_seconds / (max_rows - 1))
    interval_seconds = interval_seconds or 1 # In case all records on same day-hour-minute

    # Format interval display to numbers of days/hours/minutes
    days = interval_seconds // (60 * 60 * 24)
    seconds = interval_seconds - (days * 60 * 60 * 24)
    hours = seconds // (60 * 60)
    seconds = seconds - hours * 60 * 60
    minutes = seconds // 60
    interval_str = '{}day {}hour {}minute'.format(int(days), int(hours), int(minutes))

    df_date_cpu = interval_cpu_mean_df(df, interval_seconds)
    return (df_date_cpu, interval_seconds, interval_str)

def hosts_freqseries(df, interval_seconds=None):
    """
    Get a list of cpu value timeseries for each host that exists in
    input dataframe
    For each host dataframe, fill missing values with adjacent ones
    df: dataframe of the form
    ('date1', 'hostname1', 'os', 'cpu')
    ('date1', 'hostname2', 'os', 'cpu')
    ...
    ('date2', 'hostname1', 'os', 'cpu')
    ('date2', 'hostname2', 'os', 'cpu')
    interval_seconds: if specified, return time series averaged on the specified interval
    Return a list of time series having the form
    [
        [
            cpu, # For date1 hostname1
            cpu, # For date2 hostname1
        ],
        [
            cpu, # For date1 hostname2
            cpu, # For date2 hostname2
        ],
        ...
    ]
    """
    dfs = []
    hostnames = df.hostname.unique()
    dates = df.date.unique()
    interval = pd.Timedelta(dates[1] - dates[0]).total_seconds()
    date_range = pd.date_range(dates[0], dates[-1], freq='{}S'.format(interval))
    for hostname in hostnames:
        host_df = df.loc[df.hostname == hostname].copy()
        host_df.date = pd.to_datetime(host_df.date)
        # Set date as the index
        host_df.set_index('date', inplace=True)
        # Fill missing dates
        host_df = host_df.reindex(date_range)
        # Set values of missing dates by interpolation
        # Pad method will use adjacent string/numeric values from the previous
        # adjacent index
        host_df.interpolate(method='pad', inplace=True)
        if interval_seconds:
            # Set back index to default
            host_df = host_df.rename_axis('date').reset_index()
            host_df = interval_cpu_mean_df(host_df, interval_seconds)
        host_df = host_df[['cpu']]
        # Fillna in case first elements are None
        host_df.fillna(0, inplace=True)
        dfs.append(host_df)
    return dfs


# def truncates_time(df):
    # '''
        # truncate values of dataframe and adds column to it
    # '''
    # f = FREQ_SAMPLING[DB_QUERY_ARGS['interval']]

    # # Read & prep data
    # df['date_trunc'] = df['date'].dt.floor(f)
    # df = df.set_index('date_trunc')
    # return df

# def prepare_data(df,min_max_date):
    # '''
        # min_max_date = [mindate,maxdate]

        # creates dataframe following specified limit dates. 

        # input df must have truncated dates and correspond to a single host.

        # if date is not present in the original dataframe it
        # paddes with zeros
    # '''
    # f = FREQ_SAMPLING[DB_QUERY_ARGS['interval']]


    # # Cpu value per hour
    # counts = df[['cpu']]

    # # Create time & signal, filling in missing hours with 0 calls
    # counts_dict = counts['cpu'].to_dict()
    # time = pd.date_range(min_max_date[0], min_max_date[1], freq = f).to_series().sort_values()
    # signal = time.apply(lambda x: counts_dict[x] if x in counts_dict.keys() else 0)

    # # Set signal to be the difference of call volume from the average
    # signal = signal - signal.mean()

    # signal = pd.DataFrame({'date':signal.index, 'cpu': signal.values })

    # return signal

# def hosts_timeseries_for_fourier(df):
    # '''
        # df received is the one with all alias together.
        # Returns signal matrix, a list of time series dataframes from
        # each host, all having same size and equally time 
        # spanned.
    # '''
    
    # df = truncates_time(df)
    
    # min_max_date = [min((df[['cpu']]).index),max((df[['cpu']]).index)]
    
    # signal_matrix = []
    # for alias in df.hostname.unique():
        # signal =  prepare_data(df[df.hostname==alias],min_max_date)
        # signal_matrix.append(signal)
    # return signal_matrix   


