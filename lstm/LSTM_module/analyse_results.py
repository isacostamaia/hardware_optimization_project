from os import stat
import numpy as np
import pandas as pd


def stat_df():
    '''
        deprecated
    '''

    dataframe = pd.read_csv("./lstm/My_loop.txt",sep=', ', engine='python', names = ['model_version', 'machine', 'dtw_mean_fig', 'mixed_error_fig'])

    stat_df = dataframe.copy().replace([np.inf, -np.inf], np.nan)

    stat_df_dtw_ = stat_df.groupby('model_version').agg({'dtw_mean_fig':['mean', 'std']})
    stat_df_dtw_.columns = ["_".join(x) for x in stat_df_dtw_.columns.ravel()]
    stat_df_dtw_ = stat_df_dtw_.sort_values(by='dtw_mean_fig_mean')

    stat_df_mixed_ = stat_df.groupby('model_version').agg({'mixed_error_fig':['mean', 'std']})
    stat_df_mixed_.columns = ["_".join(x) for x in stat_df_mixed_.columns.ravel()]
    stat_df_mixed_ = stat_df_mixed_.sort_values(by='mixed_error_fig_mean')

    print(stat_df_dtw_)
    print(stat_df_mixed_)
    
    return stat_df_dtw_, stat_df_mixed_

def read_global_metrics():
    '''
        read My_loop_global_var_and_metrics.txt generated in train_and_predict and return dataframe
    '''

    columns_names = ['model_version', 'hostname', 'num_weeks', 'start_date', 'criterion', 'seq_len', 'learning_rate', 'epochs', 'num_layers', 'batch_size', 'hidden_size', 'dropout', 'duration_train_min',  'dtw', 'mixed_err']

    dataframe = pd.read_csv("./lstm/My_loop_global_var_and_metrics.txt",sep=', ', header=None, index_col=False, engine='python', names= columns_names)

    dataframe = dataframe.groupby('model_version').agg({'dtw':['mean', 'std'],
                                            'mixed_err':['mean', 'std'],
                                            'num_weeks':(lambda x: x.unique()),
                                            'criterion':(lambda x: x.unique()),
                                            'seq_len':(lambda x: x.unique()),
                                            'learning_rate':(lambda x: x.unique()),
                                            'epochs':(lambda x: x.unique()),
                                            'num_layers':(lambda x: x.unique()),
                                            'batch_size':(lambda x: x.unique()),
                                            'hidden_size':(lambda x: x.unique()),
                                            'dropout':(lambda x: x.unique()),
                                            'duration_train_min':(lambda x: x.unique()),
                                        })
    dataframe.columns = ["_".join(x) for x in dataframe.columns[:4].ravel()] + [x[0] for x in dataframe.columns[4:].ravel()]
    dataframe = dataframe.sort_values(by='dtw_mean')
    print(dataframe)
    return dataframe

# Run as program
if __name__ == '__main__':
#    read_global_metrics() 
    read_global_metrics()
    # stat_df_dtw_, stat_df_mixed_ = stat_df()
    # stat_df_dtw_.to_csv("./lstm/stat_df_dtw.csv")
    # stat_df_mixed_.to_csv("./lstm/stat_df_mixed.csv")