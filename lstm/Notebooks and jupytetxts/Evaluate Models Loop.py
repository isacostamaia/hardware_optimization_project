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


# +
columns_names = ['model_version', 'hostname', 'num_weeks', 'start_date', 'criterion', 'seq_len', 'learning_rate', 'epochs', 'num_layers', 'batch_size', 'hidden_size', 'dropout', 'duration_train_min',  'dtw', 'mixed_err']

dataframe = pd.read_csv("../lstm/My_loop_global_var_and_metrics.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

dataframe
# -

dataframe.criterion

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

dataframe = dataframe.sort_values(by='dtw_mean')
dataframe = dataframe.reset_index()
dataframe

dataframe.criterion.unique()


