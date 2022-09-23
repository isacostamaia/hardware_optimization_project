# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from datetime import datetime, timedelta

import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df


# +
# # %%cmd
# env\scripts\activate
# pip install kneed
# -

# ### Fetch DB records

# +
# %%time

LIST_MACH = [
'dell3353dsy',
'dell3355dsy',
'dell3356dsy',
'dell3357dsy',
'dell3358dsy',
'dell3359dsy',
'wip278dsy',
'wip261dsy',
'wip279dsy',
'wip277dsy',
'pool28dsy',
'pool29dsy',
'pool31dsy',
'pool36dsy',
'client14dsy',
'client14xdsy',
'enojarnacdsy',
'enoprj2016dsy'

]

LIST_MACH_2 = [

'ssdhwip008dsy',
'ssdhwip007dsy',
'dell3353dsy',
'dell3354dsy',
'dell3355dsy',
'dell3356dsy',
'dell3357dsy',
'dell3358dsy',
'dell3359dsy',
'dell3360dsy',
'dell3361dsy',
'dell3362dsy', #no data from this one
'dell3364dsy',
'dell3365dsy',
'dell3366dsy',
'dell3367dsy',
'dell3368dsy',
'dell3369dsy',
'ENOWINB001DSY', 
'ENOWINB002DSY', 
'WIP285DSY',
'WIP311DSY',
'WIP326DSY',
'WIP327DSY',
'WIP328DSY',
'WIP329DSY',
'WIP330DSY',
'WIP331DSY',
'WIP332DSY',
'WIP333DSY',
'WIP334DSY',
'WIP339DSY',
'WIP340DSY',
'WIP342DSY',
'WIP344DSY',
]

# Set interval & filters
query_params = {
    # 'day', 'hour', 'minute'
    'interval': 'hour',
    # datetime
    'start_date': datetime.now() - timedelta(days=7),
    # datetime
    'end_date': None,
    # 'windows', 'linux'
    'os': None,
    # List of host names
    'machines_to_include': LIST_MACH_2,
    # List of host names
    'machines_to_exclude': None,
    # Max number of records to fetch
    'limit': None
}

query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
records = query.all()
# -

df = pd.DataFrame(records, columns=['start_time', 'alias', 'os', 'per_CPU_use'])

# Read & prep data
df['timestamp_trunc'] = df['start_time'].dt.floor('H')
df = df.set_index('timestamp_trunc')

df


# +
def clustering(X, n_clusters):
    model =  KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return model

def elbow(X):
    inertia_list = []
    K = range(2,10)
    for i in K:
        model = clustering(X,i)
        inertia_list.append(model.inertia_)
    
    
    f = plt.figure(figsize=(16,8))
    plt.plot(K, inertia_list, 'bx-')
        
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    
    try:
        #knee
        kn = KneeLocator(K, inertia_list, curve='convex', direction='decreasing')
        #plot knee 
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    except:
        print("Could not compute elbow")

X = df.per_CPU_use.values.reshape(-1, 1)
elbow(X)
# -

model = clustering(X,4)
results = df.copy()
results['y_pred'] = model.labels_
results

results[results['y_pred']==0]


# +
def color_(x):
    if x==2:
        return 'm'
    if x==0:
        return 'g'
    if x==1:
        return 'r'
    if x==3:
        return 'y'
    
results['color'] = [color_(x) for x in results.y_pred]

# +
all_alias = results.alias.unique()
# all_alias = all_alias[:int(len(all_alias)/3)]

def plot_activity_clustering(all_alias):
    fig,axs = plt.subplots(len(all_alias),1,figsize=(20,20),sharex=True, sharey=True)
    for ax,alias in zip(axs,all_alias):
        df_a = results[results.alias==alias]
        ax.plot(df_a.index,df_a.per_CPU_use,label=alias, c = 'black')
        ax.scatter(df_a.index,df_a.per_CPU_use,  c = df_a.color,label=alias, s=200)
        ax.set_ylim([0,75])
        ax.set_title(alias)

    fig.subplots_adjust(hspace = 0.8)

plot_activity_clustering(all_alias[:int(len(all_alias)/3)])
# -
plot_activity_clustering(all_alias[int(len(all_alias)/3):int(2*len(all_alias)/3)])


plot_activity_clustering(all_alias[int(2*len(all_alias)/3):])




