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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# +
columns_names = ['model_version', 'batch_size', 'criterion', 'dropout', 'epochs', 'hidden_size', 'learning_rate', 'mean_dtw', 'mean_mixed_err', 'num_layers', 'seq_len', 'start_date']

dataframe = pd.read_csv("My_loop_global_var_and_metrics_2.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

# -

dataframe = dataframe.drop(['start_date'], axis=1)
dataframe

# +
sns.set_theme(style="ticks")


pp = sns.pairplot(data=dataframe,
                  y_vars=['mean_dtw'],
                  x_vars=['model_version','batch_size', 'criterion', 'dropout', 'epochs',
                           'hidden_size', 'learning_rate', 
                           'num_layers', 'seq_len'])
pp.fig.set_size_inches(20,7)

# -

plt.figure(figsize=(20,5))
plt.plot(dataframe.model_version, dataframe.mean_dtw) #, cmap = dataframe.criterion , colors = ['r', 'g']
plt.plot(dataframe.model_version, dataframe.batch_size)
plt.plot(dataframe.model_version, dataframe.learning_rate*30000)
plt.plot(dataframe.model_version, dataframe.hidden_size/100)
plt.plot(dataframe.model_version, dataframe.num_layers)

# +
var = ['mean_dtw','batch_size', 'criterion', 'dropout', 'epochs',
                           'hidden_size', 'learning_rate', 
                           'num_layers', 'seq_len']
a = 0.05
fig = make_subplots(rows=len(var), cols=2, row_heights=[0.6, a, a, a, a, a, a, a, a])

for i, v in enumerate(var): 
    fig.add_trace(
        go.Scatter(x=dataframe.model_version, y=dataframe[v], name = v),
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

dataframe[dataframe.mean_dtw == np.min(dataframe.mean_dtw)]


