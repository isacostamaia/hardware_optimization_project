"""
Run Nagios CPU usage machines
"""
import os
from datetime import date, datetime
import random
import json

from .settings import (
    DB_QUERY_ARGS,
    NUMBER_OF_RANDOM_MACHINES,
    CPU_THRESHOLD_NORMALIZATION,
    INTERVAL_SECONDS,
    TSKM_ARGS,
    RESULTS_DIRECTORY
)
from .data_source.db_query import get_cpu_query
from .data_source.dataframe import db_to_df, hosts_timeseries, normalized_cpu_threshold_df
from .clustering.clustering import run_clustering
from .outputs.outputs import write_clustering_results

def _json_default(obj):
    """
    Convert datetimes to string for json export
    """
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()

def dataset_and_clustering():
    """
    Create a filtered and formated dataset according to settings and run clustering
    """
    start = datetime.now()

    dataset_start = datetime.now()
    print('{} Fetching dataset'.format(dataset_start))
    # DB query
    query = get_cpu_query(
        **DB_QUERY_ARGS
    )
    records = query.all()

    # Convert DB records to dataframe
    initial_df = db_to_df(records)

    # Select a number of random machines if specified
    if NUMBER_OF_RANDOM_MACHINES:
        machines = random.sample(
            list(initial_df.hostname.unique()),
            k=NUMBER_OF_RANDOM_MACHINES
        )
        initial_df = initial_df[initial_df.hostname.isin(machines)]

    # Normalize CPU values against specified thresholds if specified
    if CPU_THRESHOLD_NORMALIZATION:
        normalized_cpu_df = normalized_cpu_threshold_df(
            initial_df,
            CPU_THRESHOLD_NORMALIZATION
        )
    else:
        normalized_cpu_df = initial_df

    # Get tslearn compatible timeseries, with missing dates
    # filled automatically
    # Increase time series interval if specified
    timeseries = hosts_timeseries(
        normalized_cpu_df,
        interval_seconds=INTERVAL_SECONDS
    )
    dataset_end = datetime.now()
    print('{} Dataset fetched'.format(dataset_end))
    dataset_duration = int((dataset_end - dataset_start).total_seconds())
    print('Dataset duration: {}s'.format(dataset_duration))
    print()

    clustering_start = datetime.now()
    print('{} Starting clustering'.format(clustering_start))
    tskm = run_clustering(timeseries, **TSKM_ARGS)
    clustering_end = datetime.now()
    print('{} Finished clustering'.format(clustering_end))
    clustering_duration = int((clustering_end - clustering_start).total_seconds())
    print('Clustering duration: {}s'.format(clustering_duration))
    print()

    # Convert output directory to absolute path
    results_directory = os.path.abspath(RESULTS_DIRECTORY)

    # Create output directory if not existing
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)

    results_start = datetime.now()
    print('{} Writing results to {}'.format(results_start, results_directory))
    write_clustering_results(
        initial_df,
        normalized_cpu_df,
        tskm,
        results_directory
    )
    results_end = datetime.now()
    print('{} Results written'.format(results_end))
    results_duration = int((results_end - results_start).total_seconds())
    print('Results duration: {}s'.format(results_duration))
    print()

    end = datetime.now()
    duration = int((end - start).total_seconds())
    print('Finished')
    print('Total duration: {}s'.format(duration))

    # Write summary to output directory
    summary = {
        'start_date': start,
        'settings': {
            'DB_QUERY_ARGS': DB_QUERY_ARGS,
            'NUMBER_OF_RANDOM_MACHINES': NUMBER_OF_RANDOM_MACHINES,
            'CPU_THRESHOLD_NORMALIZATION': CPU_THRESHOLD_NORMALIZATION,
            'INTERVAL_SECONDS': INTERVAL_SECONDS,
            'TSKM_ARGS': TSKM_ARGS,
            'RESULTS_DIRECTORY': results_directory,
        },
        'durations': {
            'dataset': dataset_duration,
            'clustering': clustering_duration,
            'results': results_duration,
            'total': duration,
        }
    }
    with open(os.path.join(results_directory, 'summary.json'), 'w') as summary_file:
        json.dump(summary, summary_file, sort_keys=False, indent=4, default=_json_default)

# Run as program
if __name__ == '__main__':
    dataset_and_clustering()
