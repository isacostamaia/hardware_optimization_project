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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df
from auxiliar.auxiliary_tools import save_a_cluster_small_single_alias_rolling_mean,\
                                    figures_to_html_stacked_and_side_by_side_from_folder


# +
from collections import defaultdict
import numpy as np

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette,linkage
import matplotlib.pyplot as plt
# from fastcluster import linkage
import seaborn as sns
from matplotlib.colors import rgb2hex, colorConverter
# -

# ### Fetch DB records

# +
# %%time

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

df0 = pd.DataFrame(records, columns=['start_time', 'alias', 'os', 'per_CPU_use'])

c2 = ['dell1963dsy', 'dell3353dsy', 'dell3355dsy', 'dell3356dsy',
       'dell3357dsy', 'dell3358dsy', 'dell3359dsy', 'dell3360dsy',
       'dell3361dsy', 'dell3364dsy', 'dell3365dsy', 'dell3366dsy',
       'dell3369dsy', 'dell3370dsy', 'enowinb002dsy', 'enowinb003dsy',
       'enowip01dsy', 'pool10dsy', 'pool12dsy', 'pool13dsy', 'pool23dsy',
       'pool24dsy', 'pool28dsy', 'pool31dsy', 'pool46dsy', 'pool48dsy',
       'pool52dsy', 'pool55dsy', 'pool56dsy', 'pool59dsy', 'pool60dsy',
       'ssdclient15xdsy', 'ssdhwip004dsy', 'wip178dsy', 'wip189dsy',
       'wip201dsy', 'wip285dsy', 'wip300dsy', 'wip303dsy', 'wip310dsy',
       'wip326dsy', 'wip327dsy', 'wip328dsy', 'wip330dsy', 'wip331dsy',
       'wip332dsy', 'wip333dsy', 'wip334dsy', 'wip336dsy', 'wip339dsy',
       'wip342dsy', 'wip345dsy', 'wip346dsy', 'wip347dsy', 'wip348dsy',
       'wip350dsy', 'wip351dsy', 'wip352dsy', 'wip353dsy', 'wip354dsy',
       'wip370dsy', 'dell3367dsy', 'dell3368dsy', 'enowip03dsy',
       'pool49dsy', 'wip179dsy', 'wip329dsy', 'wip340dsy', 'wip344dsy',
       'wip132dsy']
df0 = df0[df0.alias.isin(c2)]

df0 = df0.groupby('alias').agg({'per_CPU_use':[('mean','mean'),('std','std')]}).dropna()
df0

#Normalizing data
x = df0.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df

X = df.to_numpy()
X

# +
labels = range(1, len(df)+1)
labels = df0.index.values
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

# +
linked = linkage(X, 'single')

labelList = range(1, len(df)+1)
labelList = df0.index.values

# plt.figure(figsize=(100, 70),dpi=200)
plt.figure(figsize=(15, 15),dpi=100)
plt.title('Dendrogram without truncation')
P = dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

# +
Z = linkage(X, 'single')
# plt.figure(figsize=(50, 35),dpi=200)
plt.figure(figsize=(20, 15))
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')

dendrogram(
    Z,
#     truncate_mode='lastp',  # show only the last p merged clusters
#     p=100,  # show only the last p merged clusters
#     show_leaf_counts=False,  # otherwise numbers in brackets are counts
    labels=labelList,
    distance_sort='descending',
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

# -

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


plt.figure(figsize=(20, 15))
fancy_dendrogram(
    Z,
#     truncate_mode='lastp',
#     p=100,
    labels=labelList,
    leaf_rotation=90.,
    distance_sort='descending',
    leaf_font_size=12.,
    show_contracted=True,
#     annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()

# +
plt.figure(figsize=(20, 15))
max_d = 0.0578
# max_d = 0.00701

dendo = fancy_dendrogram(
    Z,
#     truncate_mode='lastp',
#     p=100,
    labels=labelList,
    leaf_rotation=90.,
    distance_sort='descending',
    leaf_font_size=12.,
    show_contracted=True,
#     annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()


# +
class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'
        
        html += '</table>'
        
        return html
    
def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    
    cluster_classes = Clusters()
    dic={}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
        dic[c]=i_l

    
    return cluster_classes, dic
# -

cluster_classes, dic = get_cluster_classes(dendo)
cluster_classes

# +
#Plot those machines from C1
# -

alias_of_int = dic['C1']
len(alias_of_int)
dic.keys()

# +
df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_normalized_data_best_version/merged_df_3kmeans.csv')
df_nkmeans_group_of_int = df_nkmeans[df_nkmeans.alias.isin(alias_of_int)]
df = pd.read_csv('auxiliar/df/df_from10-04on.csv')


save_a_cluster_small_single_alias_rolling_mean(df,df_nkmeans_group_of_int,folder_name='Atemporal Clustering',note= 'Cluster1_C1')

figures_to_html_stacked_and_side_by_side_from_folder(folder_name='Atemporal Clustering/Cluster1_C1', filename='Atemporal Clustering/Cluster1_C1.html')
# +
#plot machines from all clusters 

# +
df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_normalized_data_best_version/merged_df_3kmeans.csv')

df = pd.read_csv('auxiliar/df/df_from10-04on.csv')

# +
for key in dic.keys():
    alias_of_int = dic[key]
    
#     df_nkmeans_group_of_int = df_nkmeans[df_nkmeans.alias.isin(alias_of_int)]
#     save_a_cluster_small_single_alias_rolling_mean(df,df_nkmeans_group_of_int,folder_name='Atemporal Clustering_discretizing_cluster_2',note= 'Cluster2_{0}'.format(key))
    figures_to_html_stacked_and_side_by_side_from_folder(folder_name='Atemporal Clustering_discretizing_cluster_2/Cluster2_{0}'.format(key), filename='Atemporal Clustering_discretizing_cluster_2/Cluster2_{0}.html'.format(key))
# -



