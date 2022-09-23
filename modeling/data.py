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
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import preprocessing


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df

# -

# ### Fetch DB records

# +
# %%time

from datetime import datetime, timedelta

import pandas as pd

from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df
list_mach = ['wip196dsy','ssdhwip017dsy','wip212dsy','wip214dsy','pool27dsy','pool36dsy']

# Set interval & filters
query_params = {
    # 'day', 'hour', 'minute'
    'interval': 'minute',
    # datetime
    'start_date': datetime.now() - timedelta(weeks =1),
    # datetime
    'end_date': None,
    # 'windows', 'linux'
    'os': None,
    # List of host names
    'machines_to_include': None,
    # List of host names
    'machines_to_exclude': None,
    # Max number of records to fetch
    'limit': None
}

query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
records = query.all()




# ### Load in dataframe
# -

df = pd.DataFrame(records, columns=['start_time', 'alias', 'os', 'per_CPU_use'])

df = df.groupby('alias').agg({'per_CPU_use':[('sum','sum'),('std','std')]}).dropna()
df

#Normalizing data
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df

X = df.to_numpy()
X

# +
labels = range(1, len(df)+1)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
# -

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# +
linked = linkage(X, 'single')

labelList = range(1, len(df)+1)

plt.figure(figsize=(200, 140))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
# -

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
len(cluster.fit_predict(X))

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')

cluster2 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster2.fit_predict(X)
plt.scatter(X[:,0],X[:,1], c=cluster2.labels_, cmap='rainbow')

# ### analyse

len(df[cluster2.labels_==2])

a1 = list(df[cluster2.labels_==2].index.values)
len(a1)

df_nkmeans = pd.read_csv('Results/Clustering results-windows_nagios_all_data/df_3kmeans.csv')

a1_ = list(df_nkmeans[df_nkmeans.y_pred==0].alias.values)
len(a1_)

list1_as_set = set(a1)
intersection = list1_as_set.intersection(a1_)
len(intersection)

# +
###### c2
# -

a2 = list(df[cluster2.labels_==0].index.values)
len(a2)

a2_ = list(df_nkmeans[df_nkmeans.y_pred==1].alias.values)
len(a2_)

list1_as_set = set(a2)
intersection = list1_as_set.intersection(a2_)
len(intersection)


