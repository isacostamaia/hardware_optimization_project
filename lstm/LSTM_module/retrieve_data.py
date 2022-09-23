
import random
from datetime import date, datetime

from .data_source.db_query import get_cpu_query
from .data_source.dataframe import db_to_df, hosts_timeseries, normalized_cpu_threshold_df
from . import settings


def retrieve_data():
    '''
    Retrieves data from dataset as in settings 
    and time taken to permor retrieval
    '''

    dataset_start = datetime.now()
    print('{} Fetching dataset'.format(dataset_start))

    # DB query
    query = get_cpu_query(
        **settings.DB_QUERY_ARGS
    )
    records = query.all()

    # Convert DB records to dataframe
    initial_df = db_to_df(records)

    dataset_end = datetime.now()
    print('{} Dataset fetched'.format(dataset_end))
    dataset_duration = int((dataset_end - dataset_start).total_seconds())
    print('Dataset duration: {}s'.format(dataset_duration))
    print()

    return initial_df, dataset_duration


def input_df(initial_df):
    '''
    Given the retrieved data from database, filters and or 
    transofrm data as it will be used in clustering
    '''

    # Normalize CPU values against specified thresholds if specified
    if settings.CPU_THRESHOLD_NORMALIZATION:
        normalized_cpu_df = normalized_cpu_threshold_df(
            initial_df,
            settings.CPU_THRESHOLD_NORMALIZATION
        )
    else:
        normalized_cpu_df = initial_df
    
    return normalized_cpu_df

def retrieve_series(normalized_cpu_df):

    # Get tslearn compatible timeseries, with missing dates
    # filled automatically
    # Increase time series interval if specified
    timeseries, hostnames = hosts_timeseries(
        normalized_cpu_df,
        interval_seconds=settings.INTERVAL_SECONDS
    )
    print("Time Series Computed")
    # print("Shape timeseries: ", timeseries.shape)
    return timeseries, hostnames

def get_df():
    '''
        Run the three functions above.
        Retrieve df from db possibly normalized and interpolated.
    '''

    #retrieve from db
    initial_df, _ = retrieve_data()

    #normalize if necessary
    normalized_cpu_df = input_df(initial_df)

    #intepolate
    df, hostnames =  retrieve_series(normalized_cpu_df)

    return df, hostnames


 # Run as program
if __name__ == '__main__':
    print(get_df())