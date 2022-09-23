# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# # %%cmd
# pip install plotly
# -

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# +
columns_names = ['model_version', 'hostname', 'num_weeks', 'start_date', 'criterion', 'seq_len', 'learning_rate', 'epochs', 'num_layers', 'batch_size', 'hidden_size', 'dropout', 'duration_train_min',  'dtw', 'mixed_err']

dataframe = pd.read_csv("../lstm/My_loop_global_var_and_metrics.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

dataframe
# -

dataframe = dataframe.groupby('model_version').agg({
                                        'dtw':['mean', 'std'],
                                        'mixed_err':['mean', 'std'],
                                        'num_weeks': (lambda x: x.unique()),
                                        'criterion': (lambda x: x.unique()),
                                        'seq_len': (lambda x: x.unique()),
                                        'learning_rate':(lambda x: x.unique()),
                                        'epochs':       (lambda x: x.unique()),
                                        'num_layers':   (lambda x: x.unique()),
                                        'batch_size':   (lambda x: x.unique()),
                                        'hidden_size':  (lambda x: x.unique()),
                                        'dropout':      (lambda x: x.unique()),
                                        'duration_train_min':(lambda x: x.unique()),
                                    })
dataframe.columns = ["_".join(x) for x in dataframe.columns[:4].ravel()] + [x[0] for x in dataframe.columns[4:].ravel()]

dataframe = dataframe.reset_index()
dataframe

# +
sns.set_theme(style="ticks")


pp = sns.pairplot(data=dataframe,
                  y_vars=['dtw_mean'],
                  x_vars=['model_version','batch_size', 'criterion', 'dropout', 'epochs',
                           'hidden_size', 'learning_rate', 
                           'num_layers', 'seq_len'])
pp.fig.set_size_inches(20,7)

# -

plt.figure(figsize=(20,5))
plt.plot(dataframe.model_version, dataframe.dtw_mean) #, cmap = dataframe.criterion , colors = ['r', 'g']
plt.plot(dataframe.model_version, dataframe.batch_size)
plt.plot(dataframe.model_version, dataframe.learning_rate*30000)
plt.plot(dataframe.model_version, dataframe.hidden_size/100)
plt.plot(dataframe.model_version, dataframe.num_layers)

plt.figure(figsize=(20,5))
plt.plot(dataframe.index, dataframe.dtw_mean) #, cmap = dataframe.criterion , colors = ['r', 'g']
plt.plot(dataframe.index, dataframe.batch_size)
plt.plot(dataframe.index, dataframe.learning_rate*30000)
plt.plot(dataframe.index, dataframe.hidden_size/100)
plt.plot(dataframe.index, dataframe.num_layers)

# +
var = ['dtw_mean','batch_size', 'criterion', 'dropout', 'epochs',
                           'hidden_size', 'learning_rate', 
                          'num_layers', 'seq_len', 'num_weeks']
size_dtw = 0.6
a = (1-size_dtw)/len(var) #0.05
row_weights = [size_dtw]
[row_weights.append(a) for i in range(1, len(var))]

fig = make_subplots(rows=len(var), cols=2, row_heights=row_weights)

for i, v in enumerate(var): 
    fig.add_trace(
        go.Scatter(x=dataframe.index, y=dataframe[v], name = v),
        row=i+1, col=1
    )

fig.update_layout(height=900, width=3500, title_text="DTW distance and variables vs. model version") #700,3000
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.45
))
fig.show()

fig.write_html('visualization.html', auto_open = True)

# -

dataframe[dataframe.dtw_mean == np.min(dataframe.dtw_mean)]

dataframe = dataframe.sort_values(by='dtw_mean').reset_index().drop(['index'], axis=1)
dataframe.head(20)

[print(dataframe.head(20)[a].value_counts()/len(dataframe.head(20)), "\n\n") for a in dataframe.drop([
                                    'model_version',
                                    'dtw_mean',
                                    'mean_mixed_err'], axis=1).columns]

[print(dataframe[(dataframe.criterion!='SoftDTW(gamma=1.0)') ].head(20)[a].value_counts()/len(dataframe.head(20)), "\n\n") for a in dataframe.drop([
                                    'model_version',
                                    'dtw_mean',
                                    'mean_mixed_err'], axis=1).columns]

[print(dataframe[(dataframe.criterion!='SoftDTW(gamma=1.0)') & (dataframe.dropout != 0)].head(20)[a].value_counts()/len(dataframe.head(20)), "\n\n") for a in dataframe.drop([
                                    'model_version',
                                    'dtw_mean',
                                    'mean_mixed_err'], axis=1).columns]

# +
#another approach

top = dataframe.head(20)
param = dataframe.drop(['model_version',
                            'dtw_mean',
                            'mean_mixed_err'], axis=1).columns
for p in param:
    for v in dataframe[p].unique():
        print('The dtw mean for {} = {} is {}'.format(p,v,dataframe[dataframe[p]==v].dtw_mean.mean()))
    print('\n\n')
# -

###try derivative
col = df.drop(['model_version',
                            'dtw_mean',
                            'mean_mixed_err'], axis=1).columns
df.groupby(list(col)).filter(lambda g: len(g) > 1).drop_duplicates(subset=list(col), keep="first")

df.groupby(list(col))

# ## With other loss

# +
columns_names = ['model_version', 'batch_size', 'criterion', 'dropout', 'epochs', 'hidden_size', 'learning_rate', 'dtw_mean', 'mean_mixed_err', 'num_layers', 'seq_len', 'start_date']

df = pd.read_csv("My_loop_global_var_and_metrics_3.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

df = df.sort_values(by='dtw_mean').reset_index().drop(['index'], axis=1)


df.drop_duplicates(subset ="model_version",
                     keep = 'first', inplace = True)

# df.head(20).drop(['model_version',
#                          'dtw_mean',
#                          'mean_mixed_err'], axis=1)
# -

[print(df.head(20)[a].value_counts()/len(df.head(20)), "\n\n") for a in df.drop([
                                    'model_version',
                                    'dtw_mean',
                                    'mean_mixed_err'], axis=1).columns]

dataframe[dataframe.criterion != 'SoftDTW(gamma=1.0)'].head(20).drop([
#                         'model_version',
#                          'dtw_mean',
                         'mean_mixed_err'], axis=1)









df[df.criterion == 'SoftDTW(gamma=1.0)'].model_version.sort_values()

df.criterion.value_counts()


