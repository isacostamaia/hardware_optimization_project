"""
Time series clustering function
"""
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

def run_clustering(dataset, **kwargs):
    """
    Run the clustering function
    dataset: a list of time series values (one series per host)
    (0, 1, 5, 2),
    (17, 20, 12, 15),
    (0, 1, 0, 1)
    Return the clustering function result object
    """
    formated_dataset = to_time_series_dataset(dataset)
    tskm = TimeSeriesKMeans(**kwargs)
    tskm.fit_predict(formated_dataset)
    return tskm
