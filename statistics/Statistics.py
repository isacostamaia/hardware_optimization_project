import pandas as pd

from auxiliar import auxiliary_tools


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   df_nkmeans_not_roll = pd.read_csv('results/Cluster1/Clustering results-Cluster1_not_rolling_window/df_2kmeans.csv')
   df_nkmeans_with_roll = pd.read_csv('results/Cluster1/Clustering results-Cluster1_with_rolling_mean/df_2kmeans.csv')
   df_sv = pd.read_csv('auxiliar/df/df_sv.csv')

   # auxiliary_tools.plot_examples_rolling_mean(df_nkmeans_not_roll,df_sv,5,note='teste')

   #  auxiliary_tools.plot_examples_rolling_mean(df_nkmeans_with_roll,df_sv,10,note='Rolling Mean_Clustering cluster1 - with rolling mean')
   auxiliary_tools.plot_all_machines_rolling_mean(df_nkmeans_not_roll,df_sv,10,note='Clustering cluster1 - without rolling mean')
   auxiliary_tools.plot_all_machines_rolling_mean(df_nkmeans_with_roll,df_sv,10,note='Clustering cluster1 - with rolling mean')