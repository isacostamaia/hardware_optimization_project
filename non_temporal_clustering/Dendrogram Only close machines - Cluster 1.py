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

close = ['client14dsy', 'client14xdsy', 'cmibuild1dsy', 'dell3354dsy',
       'dell3356dsy', 'dell3359dsy', 'dell3360dsy', 'dell3365dsy',
       'dell3366dsy', 'dell3369dsy', 'eno24adsy', 'eno24bdsy',
       'eno24cdsy', 'enojarnacdsy', 'enoprj2016dsy', 'enowip01dsy',
       'enowip02dsy', 'enowip03dsy', 'enowip04dsy', 'enowip05dsy',
       'enowip06dsy', 'enowip07dsy', 'enowip10dsy', 'enowip11dsy',
       'enowip12dsy', 'enowip13dsy', 'enowip14dsy', 'enowip15dsy',
       'infra14xdsy', 'infra15dsy', 'pool08dsy', 'pool09dsy', 'pool10dsy',
       'pool11dsy', 'pool14dsy', 'pool15dsy', 'pool23dsy', 'pool24dsy',
       'pool26dsy', 'pool33dsy', 'pool38dsy', 'pool40dsy', 'pool44dsy',
       'pool45dsy', 'pool47dsy', 'pool50dsy', 'pool51dsy', 'pool53dsy',
       'ssdclient15xdsy', 'ssdhwip003dsy', 'ssdhwip006dsy',
       'ssdhwip007dsy', 'ssdhwip009dsy', 'ssdhwip010dsy', 'ssdhwip011dsy',
       'ssdhwip012dsy', 'ssdhwip013dsy', 'ssdhwip014dsy', 'ssdhwip015dsy',
       'ssdhwip016dsy', 'ssdhwip017dsy', 'ssdhwip018dsy', 'ssdhwip019dsy',
       'ssdhwip020dsy', 'ssdhwip021dsy', 'ssdhwip022dsy', 'ssdhwip023dsy',
       'ssdhwip024dsy', 'ssdhwip025dsy', 'ssdinfra15xdsy', 'ssdwip002dsy',
       'ssdwip023dsy', 'ssdwip024dsy', 'wip093dsy', 'wip094dsy',
       'wip095dsy', 'wip096dsy', 'wip097dsy', 'wip102dsy', 'wip103dsy',
       'wip106dsy', 'wip107dsy', 'wip112dsy', 'wip113dsy', 'wip114dsy',
       'wip115dsy', 'wip117dsy', 'wip118dsy', 'wip121dsy', 'wip122dsy',
       'wip123dsy', 'wip124dsy', 'wip126dsy', 'wip128dsy', 'wip131dsy',
       'wip133dsy', 'wip134dsy', 'wip135dsy', 'wip136dsy', 'wip137dsy',
       'wip138dsy', 'wip139dsy', 'wip141dsy', 'wip142dsy', 'wip146dsy',
       'wip147dsy', 'wip148dsy', 'wip150dsy', 'wip152dsy', 'wip153dsy',
       'wip154dsy', 'wip155dsy', 'wip156dsy', 'wip160dsy', 'wip161dsy',
       'wip165dsy', 'wip167dsy', 'wip168dsy', 'wip170dsy', 'wip173dsy',
       'wip174dsy', 'wip176dsy', 'wip177dsy', 'wip179dsy', 'wip181dsy',
       'wip182dsy', 'wip183dsy', 'wip184dsy', 'wip185dsy', 'wip186dsy',
       'wip187dsy', 'wip188dsy', 'wip189dsy', 'wip192dsy', 'wip193dsy',
       'wip196dsy', 'wip197dsy', 'wip198dsy', 'wip199dsy', 'wip200dsy',
       'wip203dsy', 'wip204dsy', 'wip205dsy', 'wip207dsy', 'wip208dsy',
       'wip216dsy', 'wip220dsy', 'wip222dsy', 'wip230dsy', 'wip243dsy',
       'wip244dsy', 'wip245dsy', 'wip246dsy', 'wip247dsy', 'wip248dsy',
       'wip249dsy', 'wip250dsy', 'wip251dsy', 'wip255dsy', 'wip259dsy',
       'wip305dsy', 'wip311dsy', 'wip312dsy', 'wip313dsy', 'wip314dsy',
       'wip315dsy', 'wip317dsy', 'wip318dsy', 'wip319dsy', 'wip320dsy',
       'wip322dsy', 'wip323dsy', 'wip328dsy', 'wip333dsy', 'wip335dsy',
       'wip336dsy', 'wip337dsy', 'wip338dsy', 'wip343dsy', 'wip345dsy',
       'wip346dsy', 'wip350dsy', 'wip352dsy', 'wip355dsy', 'wip356dsy',
       'wip357dsy', 'wip358dsy', 'wip359dsy', 'wip363dsy']
df0 = df0[df0.alias.isin(close)]

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
    truncate_mode='lastp',  # show only the last p merged clusters
    p=100,  # show only the last p merged clusters
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
    truncate_mode='lastp',
    p=100,
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
max_d = 0.067
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

# +
df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_normalized_data_best_version/merged_df_3kmeans.csv')
df_nkmeans_group_of_int = df_nkmeans[df_nkmeans.alias.isin(alias_of_int)]
df = pd.read_csv('auxiliar/df/df_from10-04on.csv')


save_a_cluster_small_single_alias_rolling_mean(df,df_nkmeans_group_of_int,folder_name='Atemporal Clustering',note= 'Cluster1_C1')

figures_to_html_stacked_and_side_by_side_from_folder(folder_name='Atemporal Clustering/Cluster1_C1', filename='Atemporal Clustering/Cluster1_C1.html')
# -


