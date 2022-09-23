"""
Default settings
Can be overriden by a local settings_local.py file (gitignored)
Import settings module instead of settings_default or settings_local directly
"""
import os
from datetime import datetime

#### DB settings

# Nagios database connection string
# http://lin033dsy/nagios/
DB_CONNECTION_STRING = 'mysql+pymysql://ndoutils:nagiosadmin@lin033dsy/nagios'

#### Dataset settings

# Options to pass to DB query function
DB_QUERY_ARGS = dict(
    connection_string=DB_CONNECTION_STRING,
    interval='hour',
    start_date=datetime(2021, 4, 12),
    end_date=datetime(2021, 4, 19),
    os='windows',
    machines_to_include=None,
    machines_to_exclude=None,
    limit=None
)

# If specified, length of a random subset of machine from above
# DB query result, all machines from query result otherwise
NUMBER_OF_RANDOM_MACHINES = None

# CPU values normalization thresholds
# Eg, if set to [10, 30, 60, 100], with input series
# (7, 12, 30, 55, 80, 90, 100) would get normalized to
# (0, 1, 1, 2, 3, 3, 3)
# Not applied if set to None
CPU_THRESHOLD_NORMALIZATION = [5, 30, 60, 100]

# If specified, increases interval between records to the specified
# interval, with CPU values averaged accordingly
INTERVAL_SECONDS = None

#### Clustering settings

# Options to pass to TimeSeriesKMeans
TSKM_ARGS = dict(
    n_clusters=3,
    #random_state=0,
    metric='euclidean', # 'dtw'
    n_init=4,
    max_iter=20,
    #max_iter_barycenter=50,
    n_jobs=-1
)

#### Output files settings

# Directory to store html/json/text files results
# Absolute or relative
# Created if not existing
# Note: tmp directory is gitignored
RESULTS_DIRECTORY = os.path.join('tmp', datetime.now().strftime('%Y-%m-%d_%H-%M-%S_clustering'))
