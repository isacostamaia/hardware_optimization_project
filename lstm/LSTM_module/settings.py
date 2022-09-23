import os
from datetime import datetime, timedelta

from torch import nn
from soft_dtw_cuda import SoftDTW
from pytorch_forecasting.metrics import MAPE


LOOP_NAME = "My_loop"

##VERSION: folder name. Results stored in ./lstm/[version]
VERSION = '12'

#### DB settings

# Nagios database connection string
# http://lin033dsy/nagios/
DB_CONNECTION_STRING = 'mysql+pymysql://ndoutils:nagiosadmin@lin033dsy/nagios'

#### Dataset settings

#date limits dataset
NUM_WEEKS = 9
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(weeks = NUM_WEEKS)

# Options to pass to DB query function
DB_QUERY_ARGS = dict(
    connection_string=DB_CONNECTION_STRING,
    interval='hour',
    start_date=  START_DATE, #datetime.fromisoformat('2021-03-25 11:00:00')
    end_date=  END_DATE, # datetime.fromisoformat('2021-05-27 10:00:00') 
    os='windows',
    machines_to_include= None, #['client14dsy', 'client14xdsy'],
    machines_to_exclude= ['wip132dsy', 'ssdwip010dsy', 'ssdwip017dsy', 'ssdwip018dsy', 'ssdwip020dsy', 'ssdwip021dsy', 'ssdwip022dsy'],
    limit=None
)

# If specified, increases interval between records to the specified
# interval, with CPU values averaged accordingly
INTERVAL_SECONDS = None

# CPU values normalization thresholds
# Eg, if set to [10, 30, 60, 100], with input series
# (7, 12, 30, 55, 80, 90, 100) would get normalized to
# (0, 1, 1, 2, 3, 3, 3)
# Not applied if set to None
CPU_THRESHOLD_NORMALIZATION = None # [5, 30, 60, 100] 


PARAM = dict(
    seq_len = 24,
    batch_size = 1, 
    criterion = MAPE(), #nn.MSELoss(),
    max_epochs = 10,
    # n_features = 312, #change done: depending of the data retrieved we have more or less machines. In ouput of lstm also.
    hidden_size = 200,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)

#specify if GPU is desired
GPU_DESIRED = True
