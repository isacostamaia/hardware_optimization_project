import random
from datetime import datetime, timedelta

import base64
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from kaleido.scopes.plotly import PlotlyScope
from sklearn import preprocessing

from df_alias_all import make_df_all_alias, retrieve_machines_identity
from df_sv import get_treated_df_sv,save_df_sv
from auxiliary_tools import figures_to_html, plot_list_machines,plot_small_single_alias, \
    figures_to_html_side_by_side,figures_to_html_stacked_and_side_by_side, plot_small_single_alias_rolling_mean,\
    figures_to_html_stacked_and_side_by_side_from_folder,save_all_small_single_alias_rolling_mean


from dataset.settings import DB_CONNECTION_STRING
from dataset.data_source.db_query import get_cpu_query
from dataset.data_source.dataframe import auto_interval_cpu_mean_df




class Clustering(object):
    def __init__(self,df,os,nclusters,note,sz_rolling_mean=None,norm='global'):
         
        self.df = df.loc[df.os==os]
        self.formatted_dataset = None
        self.y_pred = []
        self.dba_km = None
        self.nclusters = nclusters
        self.os=os
        self.all_alias = None 
        self.df_nkmeans = None
        self.machines_identity_os = None
        self.note=note
        self.col_name=None
        self.sz_rolling_mean = sz_rolling_mean
        self.df_input = None
        self.norm = norm

    def set_all_alias_os(self):
        frame = retrieve_machines_identity()
        self.machines_identity_os=frame.loc[frame.os==self.os]
        # self.all_alias = self.machines_identity_os.alias.values
        self.all_alias = self.df.alias.unique()

    def gen_formatted_dataset(self,sz=None):

        if ('per_CPU_use' in (self.df.columns)):
            self.col_name = 'per_CPU_use'
        elif ('avg_cpu_use' in (self.df.columns)):
            self.col_name = 'avg_cpu_use'
        else:
            return ('column of cpu use not found')


        if (self.norm=='global'):
            #Normalizing data
            print('Normalizing data...')
            x = self.df['per_CPU_use'].values.reshape(-1, 1) #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df_norm = pd.DataFrame(x_scaled)
            df_norm['alias'] = self.df.alias
            df_norm['start_time'] = df.start_time
            df_norm.rename(columns={0:'per_CPU_use'}, inplace=True) #normalized dataframe
            self.df_input = df_norm

        if (self.norm=='none'):
            self.df_input = self.df

        
        self.set_all_alias_os()

        all_time_series = [self.df_input[self.df_input.alias==alias][self.col_name].values for alias in self.all_alias]

        print(self.all_alias)

        if(self.sz_rolling_mean):
            print('Computing rolling mean with window size {0}'.format(self.sz_rolling_mean))
            all_time_series = [pd.DataFrame(a).rolling(window=self.sz_rolling_mean).std().iloc[self.sz_rolling_mean-1:].values.ravel() for a in all_time_series]

        formatted_dataset = to_time_series_dataset(all_time_series)

        # print('Normalizing data...')
        # TimeSeriesScalerMinMax().fit_transform(formatted_dataset)

        # if(sz):
        #     formatted_dataset = TimeSeriesResampler(sz=sz).fit_transform(formatted_dataset)

        self.formatted_dataset = formatted_dataset
        print('dataset formated')


    def clustering(self,njobs):
        seed = 0
        np.random.seed(seed)

        # DBA-k-means
        print("DBA k-means")
        self.dba_km = TimeSeriesKMeans(n_clusters=self.nclusters,
                                n_init=2,
                                metric="dtw",
                                verbose=True,
                                max_iter_barycenter=10,
                                random_state=seed,
                                n_jobs=njobs)

        self.y_pred = self.dba_km.fit_predict(self.formatted_dataset)

        #save model
        direc = 'Results/Clustering results-'+str(self.note)
        Path(direc).mkdir(parents=True, exist_ok=True)
        filename = direc +'/model.json'
        self.dba_km.to_json(filename)
    
        print("clustering completed")

    def retrieve_inertia(self):
        return self.df_nkmeans.inertia_

    def save_clustering_results(self):
        
        df_nkmeans = pd.DataFrame({'alias': self.all_alias, 'y_pred':self.y_pred})

        direc = 'Results/Clustering results-'+str(self.note)
        Path(direc).mkdir(parents=True, exist_ok=True)

        df_nkmeans.to_csv(r'%s/df_%dkmeans.csv'%(direc,self.nclusters))

        df_nkmeans = df_nkmeans.merge(self.machines_identity_os,how = 'left', on='alias')

        df_nkmeans.fillna('other',inplace=True)

        df_machines = pd.read_csv('df/Machines_Suivi.csv',sep=';')
        df_machines.loc[:,'Name'] = [x.lower() for x in df_machines.Name]
        df_nkmeans = df_nkmeans.merge(df_machines[['Name','Projet','Role']], how = 'left', left_on = 'alias', right_on='Name')[[
            'alias','os','level','group','y_pred','Projet','Role']]

        df_nkmeans.to_csv(r'%s/merged_df_%dkmeans.csv'%(direc,self.nclusters))

        self.df_nkmeans = df_nkmeans

        print('clustering results saved')

    def plot_cluster(self):
        '''
            plot all machines of each cluster together and display in red the centroid of that cluster. 
            Not that useful.
        '''

        sz = self.formatted_dataset.shape[1]
        for yi in range(self.nclusters):
            plt.subplots(figsize=(15,15))
            for xx in self.formatted_dataset[self.y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(self.dba_km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(0, 100)
            plt.title('DBA $k$-means -Cluster %d' % (yi + 1))
            direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
            Path(direc).mkdir(parents=True, exist_ok=True)
            plt.savefig('%s/cluster %d.png' %(direc,yi+1))
        print('cluster plot saved')
    
    def plot_per_group(self):
        color_dic = {'replay':"#1f77b4", 'other':"#ff7f0e", 'prj_build':"#2ca02c", 'preint_build':"#d62728"} #, "#9467bd", "#8c564b"
        fig, set_axes = plt.subplots(nrows=1,ncols=self.nclusters,figsize=(13,13))
        for i,ax in enumerate(set_axes): 
            d = pd.DataFrame(self.df_nkmeans[(self.df_nkmeans.y_pred==i)].group.value_counts()).reset_index()
            colors = [color_dic[r] for r in d['index']]
            ax.pie(d.group,labels=d['index'],colors=colors,normalize=True)
            ax.title.set_text('%s \n Cluster %d'%(self.os,i+1))
        direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
        Path(direc).mkdir(parents=True, exist_ok=True)
        plt.savefig('%s/%s percentage role each cluster.png' %(direc,self.os))
        print('percentage cluster plot saved')

    def plot_examples(self,n_mach_per_plot):
        '''
            make a dashboard with n_mach_per_plot plots of machines per cluster.
            Images stacked vertically.
            If rolling mean is used for clustering it is also used for displaying,
            otherwise raw values are displayed. 
            Not really interesting.
        '''

        list_fig = []
        for i in range(self.nclusters):
            #separate alias
            alias_cluster_i = self.df_nkmeans[self.df_nkmeans.y_pred==i].alias.unique()
            
            #make figure
            fig = make_subplots(rows=n_mach_per_plot,cols=1)
            for j in range(n_mach_per_plot):
                r = random.randrange(0, len(alias_cluster_i))

                if(self.sz_rolling_mean):
                    alias = alias_cluster_i[r]
                    data = (self.df[self.df.alias==alias].set_index('start_time')).loc[:,self.col_name]
                    data = pd.DataFrame(data)
                    
                    short_rolling = data.rolling(window=self.sz_rolling_mean).mean()
                    # start_date =  short_rolling.index[self.sz_rolling_mean-1]
                    # end_date = short_rolling.index[-1]
# 
                    fig.add_trace(go.Scatter(x = short_rolling.index,
                                            y=short_rolling[self.col_name],mode= 'lines', 
                                            marker=dict(line=dict(width=2,
                                                        color='yellow')), 
                                            name = alias,
                                            marker_color = 'DarkSlateGrey'
                                            ),
                                    row=j+1, col=1
                                    )
                else:
                    fig.add_trace(go.Bar(
                                        x=self.df[self.df.alias==alias_cluster_i[r]].start_time,
                                        y = (self.df[self.df.alias==alias_cluster_i[r]])[self.col_name],
                                        name = alias_cluster_i[r],
                                        marker=dict(line=dict(width=2,
                                                    color='DarkSlateGrey'))
                                        ),
                                row=j+1, col=1
                                )
            fig.update_layout(title="%s hosts classified in cluster %d using %d classes"%(self.os,i+1,self.nclusters))
                
            list_fig.append(fig)
        direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
        Path(direc).mkdir(parents=True, exist_ok=True)
        filename = "{0}/results_{1}kmeans.html".format(direc,self.nclusters)
        figures_to_html(list_fig,filename=filename)

    def plot_all_machines_rolling_mean(self):
        '''
        plot all machines in the same document using one column per cluster.
        Display the average mean using a window of 30 min.

        '''
        df = self.df_input

        axis_range_multiplier =1
        if (self.norm=='global'):   axis_range_multiplier=0.01

        list_fig = []
        for i in range(self.nclusters):
            #separate alias
            alias_cluster_i = self.df_nkmeans[self.df_nkmeans.y_pred==i].alias.unique()
            
            #make figure
            fig = make_subplots(rows=len(alias_cluster_i),cols=1)
            for r in range(len(alias_cluster_i)):
                alias = alias_cluster_i[r]
                #not done : plot only data from a week ago: &(df.start_time>=str(datetime.now()-timedelta(weeks=1))
                data = (df[(df.alias==alias)].set_index('start_time')).loc[:,self.col_name]
                data = pd.DataFrame(data)
                sz_win = 5 #20 before
                short_rolling = data.rolling(window=sz_win).mean()

                fig.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling[self.col_name],mode= 'lines', 
                                        marker=dict(line=dict(width=2,
                                                color='yellow')), 
                                        name = alias,
                                        marker_color = 'DarkSlateGrey'
                                        ),
                            row=r+1, col=1
                            )
            fig.update_yaxes(range=[0, 80*axis_range_multiplier])
            fig.update_layout(height=60*len(alias_cluster_i), width=650,title="%s hosts cluster %d using %d classes"%(self.note,i+1,self.nclusters))
            list_fig.append(fig)

        direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
        Path(direc).mkdir(parents=True, exist_ok=True)
        filename = "{0}/plot_all_machines_week_ago.html".format(direc)
        figures_to_html_side_by_side(list_fig,filename=filename)

    def plot_all_machines_one_doc_per_cluster_iteractive_plot(self):
    
        '''
            create one plot document per cluster displaying the rolling mean of each machine
            heavy visualizations
        '''
        df = self.df_input

        direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
        Path(direc).mkdir(parents=True, exist_ok=True)

        for i in range(self.nclusters):
            figs = []
            for j,row in self.df_nkmeans[self.df_nkmeans.y_pred==i].iterrows():
                figs.append(plot_small_single_alias_rolling_mean(df,row['alias'],str(row['Projet'])+' | '+str(row['Role'])))

            filename = "{0}/plot_cluster{1}.html".format(direc,i)
            figures_to_html_stacked_and_side_by_side(figs,filename)

    def plot_one_doc_per_cluster(self):

        '''
            create one plot document per cluster displaying the rolling mean of each machine
            light visualizations - Best visu method
        '''

        df = self.df_input
        axis_range_multiplier=1
        if (self.norm=='global') : axis_range_multiplier=0.01


        #save all individual figure alias
        direc = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans'
        direc_imgs = 'Results/Clustering results-'+str(self.note)+'/'+ 'Imgs '+str(self.nclusters)+'kmeans' + '/Each_alias_plot'
        Path(direc_imgs).mkdir(parents=True, exist_ok=True)
        save_all_small_single_alias_rolling_mean(df,self.df_nkmeans,direc_imgs,axis_range_multiplier,note= None)

        #make one .html file per cluster
        for i in range(self.nclusters):
            figures_to_html_stacked_and_side_by_side_from_folder('{0}/cluster_{1}'.format(direc_imgs,(i+1)), 
                                                                '{0}/cluster_{1}.html'.format(direc,(i+1)))

    def initialise(self,df_nkmeans, df):
        self.df_nkmeans = df_nkmeans
        self.df = df
        self.nclusters = len(df_nkmeans.y_pred.unique())
        self.col_name = 'per_CPU_use'
        self.df_input = df

    def gen_report(self): 
        direc = 'Results/Clustering results-'+str(self.note)+'/report_clustering.txt'
        f= open(direc,"w+")

        f.write('Made on {0} using data from {1} to {2} \r\n'.format(datetime.now(),np.min(self.df.start_time.values),np.max(self.df.start_time.values)))
        f.write('Computed with norm input mode as  {0}\r\n'.format(self.norm))   

        if(self.sz_rolling_mean):
            f.write('Computed with rolling std - window size = {0}\r\n'.format(self.sz_rolling_mean))

        for i in range(self.nclusters):
            df_cluster = self.df_nkmeans[self.df_nkmeans.y_pred==i]
            f.write("Cluster {0}- size: {1} ({2}% machines):\r\n".format(i+1,len(df_cluster),round(len(df_cluster)*100/len(self.df_nkmeans),3)))
            f.write("alias             os        level        group\r\n")
            for a,row in df_cluster.iterrows():
                f.write("%s           %s  %s  %s \r\n" % (row['alias'],row['os'],row['level'],row['group']))
            f.write("\r\n\n")

    def gen_csv_report(self):
        direc = 'Results/Clustering results-'+str(self.note)
        Path(direc).mkdir(parents=True, exist_ok=True)
        self.df_nkmeans['cluster'] = [i+1 for i in self.df_nkmeans.y_pred]
        df = pd.concat([self.df_nkmeans[self.df_nkmeans.y_pred==i] for i in range(self.nclusters)])
        df.to_csv(r'{0}/Clustering.csv'.format(direc),sep=';',encoding="utf-8",columns=['alias','os','level','group','y_pred','cluster'])



if __name__ == "__main__":
   # stuff only to run when not called via 'import' here

    # df_alias_all = pd.read_csv('df/df_alias_all.csv')
    # week_ago  = str(pd.Timestamp(datetime.now())  - timedelta(weeks=3))
    # df_alias_week_ago = df_alias_all.copy()
    # df_alias_week_ago = df_alias_week_ago.loc[df_alias_week_ago.start_time>=week_ago]

    # c = Clustering(df=df_alias_week_ago,os='windows',nclusters=3,note='windows_nagios')
    # c.gen_formatted_dataset()
    # c.clustering(njobs=-1)
    # c.save_clustering_results()
    # c.plot_cluster()
    # c.plot_per_group()

######################################################
    # df_sv = pd.read_csv('df/df_sv.csv')

    # df_3kmeans = pd.read_csv('Clustering results-windows_nagios_all_data/merged_df_3kmeans.csv')


    # plot_examples(nclusters=3,df_sv=df_sv,df_nkmeans=df_3kmeans,n_mach_per_plot=6,os='windows')

############################################################
    # list_alias = ['ssdhwip020dsy','ssdhwip015dsy','dell3370dsy', 'enojarnacdsy', 'enowip08dsy', 'pool27dsy',
    #    'pool28dsy', 'pool29dsy', 'pool30dsy', 'pool31dsy', 'pool36dsy',
    #    'pool37dsy', 'pool45dsy', 'pool53dsy', 'ssdclient15xdsy',
    #    'ssdhwip019dsy', 'wip142dsy', 'wip144dsy', 'wip178dsy',
    #    'wip205dsy', 'wip323dsy', 'wip370dsy']

    # df_sv = pd.read_csv('df/df_sv.csv')

    # plot_list_machines(df_sv,list_alias,'Timeline_Alias')
###############################################
    # df_sv = pd.read_csv('df/df_sv.csv')
    # cluster1 = ['client14dsy', 'client14xdsy', 'cmibuild1dsy', 'dell1883dsy',
    # 'dell3353dsy', 'dell3354dsy', 'dell3355dsy', 'dell3361dsy',
    # 'dell3364dsy', 'dell3365dsy', 'dell3366dsy', 'dell3369dsy',
    # 'eno24adsy', 'eno24bdsy', 'eno24cdsy', 'eno24dsy', 'enoprj2016dsy',
    # 'enowip01dsy', 'enowip02dsy', 'enowip03dsy', 'enowip04dsy',
    # 'enowip05dsy', 'enowip06dsy', 'enowip07dsy', 'enowip09dsy',
    # 'enowip10dsy', 'enowip11dsy', 'enowip12dsy', 'enowip13dsy',
    # 'enowip14dsy', 'enowip15dsy', 'infra14xdsy', 'infra15dsy',
    # 'pool08dsy', 'pool09dsy', 'pool10dsy', 'pool11dsy', 'pool12dsy',
    # 'pool13dsy', 'pool14dsy', 'pool15dsy', 'pool23dsy', 'pool24dsy',
    # 'pool26dsy', 'pool32dsy', 'pool33dsy', 'pool34dsy', 'pool35dsy',
    # 'pool38dsy', 'pool39dsy', 'pool40dsy', 'pool43dsy', 'pool44dsy',
    # 'pool46dsy', 'pool47dsy', 'pool48dsy', 'pool49dsy', 'pool50dsy',
    # 'pool51dsy', 'pool52dsy', 'pool55dsy', 'pool59dsy', 'pool60dsy',
    # 'preint2017xdsy', 'ssdhwip003dsy', 'ssdhwip006dsy',
    # 'ssdhwip009dsy', 'ssdhwip010dsy', 'ssdhwip011dsy', 'ssdhwip012dsy',
    # 'ssdhwip013dsy', 'ssdhwip014dsy', 'ssdhwip015dsy', 'ssdhwip016dsy',
    # 'ssdhwip017dsy', 'ssdhwip018dsy', 'ssdhwip020dsy', 'ssdhwip021dsy',
    # 'ssdhwip022dsy', 'ssdhwip023dsy', 'ssdhwip024dsy', 'ssdhwip025dsy',
    # 'ssdinfra15xdsy', 'ssdprj2017xdsy', 'ssdwip001dsy', 'ssdwip002dsy',
    # 'ssdwip003dsy', 'ssdwip004dsy', 'ssdwip005dsy', 'ssdwip007dsy',
    # 'ssdwip008dsy', 'ssdwip009dsy', 'ssdwip017dsy', 'ssdwip020dsy',
    # 'ssdwip021dsy', 'ssdwip022dsy', 'ssdwip023dsy', 'ssdwip024dsy',
    # 'vmyme010dsy', 'wip093dsy', 'wip094dsy', 'wip095dsy', 'wip096dsy', 
    # 'wip097dsy', 'wip102dsy', 'wip103dsy', 'wip106dsy', 'wip107dsy',
    # 'wip112dsy', 'wip113dsy', 'wip114dsy', 'wip115dsy', 'wip117dsy',
    # 'wip118dsy', 'wip121dsy', 'wip122dsy', 'wip123dsy', 'wip124dsy',
    # 'wip125dsy', 'wip126dsy', 'wip128dsy', 'wip131dsy', 'wip133dsy',
    # 'wip134dsy', 'wip135dsy', 'wip136dsy', 'wip137dsy', 'wip138dsy',
    # 'wip139dsy', 'wip141dsy', 'wip146dsy', 'wip147dsy', 'wip148dsy',
    # 'wip149dsy', 'wip150dsy', 'wip152dsy', 'wip153dsy', 'wip154dsy',
    # 'wip155dsy', 'wip156dsy', 'wip160dsy', 'wip161dsy', 'wip163dsy',
    # 'wip164dsy', 'wip165dsy', 'wip166dsy', 'wip167dsy', 'wip168dsy',
    # 'wip170dsy', 'wip173dsy', 'wip174dsy', 'wip176dsy', 'wip177dsy',
    # 'wip179dsy', 'wip181dsy', 'wip182dsy', 'wip183dsy', 'wip184dsy',
    # 'wip185dsy', 'wip186dsy', 'wip187dsy', 'wip188dsy', 'wip189dsy',
    # 'wip192dsy', 'wip193dsy', 'wip196dsy', 'wip197dsy', 'wip198dsy',
    # 'wip199dsy', 'wip200dsy', 'wip201dsy', 'wip203dsy', 'wip204dsy',
    # 'wip207dsy', 'wip208dsy', 'wip216dsy', 'wip220dsy', 'wip222dsy',
    # 'wip230dsy', 'wip243dsy', 'wip244dsy', 'wip245dsy', 'wip246dsy',
    # 'wip247dsy', 'wip248dsy', 'wip249dsy', 'wip250dsy', 'wip251dsy',
    # 'wip255dsy', 'wip257dsy', 'wip259dsy', 'wip266dsy', 'wip300dsy',
    # 'wip302dsy', 'wip305dsy', 'wip310dsy', 'wip312dsy', 'wip313dsy',
    # 'wip314dsy', 'wip315dsy', 'wip316dsy', 'wip317dsy', 'wip318dsy',
    # 'wip319dsy', 'wip320dsy', 'wip322dsy', 'wip335dsy', 'wip336dsy',
    # 'wip337dsy', 'wip338dsy', 'wip343dsy', 'wip346dsy', 'wip348dsy',
    # 'wip352dsy', 'wip354dsy', 'wip355dsy', 'wip356dsy', 'wip357dsy',
    # 'wip358dsy', 'wip359dsy', 'wip360dsy', 'wip361dsy', 'wip363dsy']


    # c = Clustering(df=df_sv,os='windows',nclusters=2,note='Cluster1_with_rolling_std',sz_rolling_mean=6)
    # print(df_sv[df_sv.alias.isin(cluster1)])
    # c.gen_formatted_dataset(2000)
    # c.clustering(njobs=-1)
    # c.save_clustering_results()
    # c.plot_cluster()
    # c.plot_per_group()
    # c.plot_examples(n_mach_per_plot = 15)
    # c.plot_all_machines_rolling_mean()
    # c.gen_report()
######################################################################
    # mach = ['dell3356dsy', 'dell3355dsy', 'client14dsy', 'client14xdsy', 'wip212dsy', 'wip219dsy']
    # # Set interval & filters
    # query_params = {
    #     # 'day', 'hour', 'minute'
    #     'interval': 'minute',
    #     # datetime
    #     'start_date': pd.Timestamp(day=10,month = 4,year=2021, hour=13),
    #     # datetime
    #     'end_date': pd.Timestamp(day=16,month = 4,year=2021, hour=13),
    #     # 'windows', 'linux'
    #     'os': None,
    #     # List of host names
    #     'machines_to_include': mach,
    #     # List of host names
    #     'machines_to_exclude': None,
    #     # Max number of records to fetch
    #     'limit': None
    # }

    # query = get_cpu_query(DB_CONNECTION_STRING, **query_params)
    # records = query.all()
    # df = pd.DataFrame(records, columns=['start_time', 'alias', 'os', 'per_CPU_use'])
    # # df.to_csv(r'df/df_from10-04on.csv')
    

    # c = Clustering(df=df,os='windows',nclusters=3,note='little_test_normalized_data')
    # c.gen_formatted_dataset(2000)
    # c.clustering(njobs=-1)
    # c.save_clustering_results()
    # c.plot_cluster()
    # c.plot_per_group()
    # # c.plot_examples(n_mach_per_plot = 15)
    # c.plot_all_machines_rolling_mean()
    # c.gen_report()
    # c.gen_csv_report()
######################################

    # df = pd.read_csv('df/df_from10-04on.csv')
    # inertia_list = []
    # K = range(1,9)
    # for i in K:
    #     c = Clustering(df=df,os='windows',nclusters=i,note='Inertia_elbow')
    #     c.gen_formatted_dataset(2000)
    #     c.clustering(njobs=-1)
    #     inertia_list.append(c.retrieve_inertia())

    # print(inertia_list)

    # plt.figure(figsize=(16,8))
    # plt.plot(K, inertia_list, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')

    # direc = 'Results/Clustering results-Inertia_elbow/elbow.png'

    # plt.savefig(direc)

    ######################################################
    df = pd.read_csv('df/df_from10-04on.csv')
    




    df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_now_normalized_correct_manner/merged_df_3kmeans.csv')

    c = Clustering(df=df,os='windows',nclusters=3,note='windows_clustering_fresh_data_not_normalized_at_all', norm='none')
    c.gen_formatted_dataset(2000)
    c.clustering(njobs=-1)
    c.save_clustering_results()
    c.plot_cluster()
    c.plot_per_group()

    # c.initialise(df_nkmeans,df)
    c.plot_all_machines_rolling_mean()
    c.plot_one_doc_per_cluster()
    c.gen_report()
    c.gen_csv_report()
