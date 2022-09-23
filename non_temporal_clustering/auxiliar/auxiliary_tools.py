import random
import math
import base64
import glob

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from kaleido.scopes.plotly import PlotlyScope

            
def figures_to_html(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")

def figures_to_html_side_by_side(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    dashboard.write("<div style='white-space:nowrap; vertical-align: top;'>")
    for fig in figs:
        dashboard.write("<div style='display: inline-block; vertical-align: top;'>")
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
        dashboard.write("</div>")
    dashboard.write("</div>")
    dashboard.write("</body></html>" + "\n")

def plot_(df_sv,list_alias,name_folder): #generate folder with a file for each machine in the list (name of file being the name of alias)
    #name_folder of type 'Machines_Folder'
    for a in list_alias:
        fig = go.Figure()
        fig.add_trace(go.Bar(
                    x=df_sv[df_sv.alias==a].start_time,
                    y = df_sv[df_sv.alias==a].per_CPU_use,
                    marker=dict( line=dict(width=2,
                                color='DarkSlateGrey')))
        )
        fig.update_layout(title="%s "%(a))

        Path(name_folder).mkdir(parents=True, exist_ok=True)
        filename = "%s/%s.html"%(name_folder,a)
        figures_to_html([fig],filename=filename)

def plot_single_alias(df_sv,alias,note= None): #plots a figure from a given alias or makes a subplot of alias if figure is given as arg
        fig = go.Figure()
        fig.add_trace(go.Bar(
                    x=df_sv[df_sv.alias==alias].start_time,
                    y = df_sv[df_sv.alias==alias].per_CPU_use,
                    marker=dict( line=dict(width=2,
                                color='DarkSlateGrey'))
                                ,
                    name=alias,
                    marker_color = 'DarkSlateGrey')
        )
        if(note):
            fig.update_layout(title="%s %s"%(alias,note))
        else:
            fig.update_layout(title="%s "%(alias))
        return fig

def plot_raw_roll_mean_single_alias(alias,df_sv,note=None):
    data = (df_sv[df_sv.alias==alias].set_index('start_time')).loc[:,'per_CPU_use']
    data = pd.DataFrame(data)
    sz_win = 20
    short_rolling = data.rolling(window=sz_win).mean()
    start_date =  short_rolling.index[sz_win-1]
    end_date = short_rolling.index[-1]

    
    f = plot_single_alias(df_sv[(df_sv.start_time<=start_date)&(df_sv.start_time>=end_date)],alias,note)
    f.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling.per_CPU_use,mode= 'lines', 
                            marker=dict(line=dict(width=2,
                                    color='yellow')), 
                            name = 'rolling mean'
                                    ))
    return f

def plot_examples(df_nkmeans,df,n_mach_per_plot,note):
    list_fig = []
    nclusters=len(df_nkmeans.y_pred.unique())
    for i in range(nclusters):
        #separate alias
        alias_cluster_i = df_nkmeans[df_nkmeans.y_pred==i].alias.unique()
        
        #make figure
        fig = make_subplots(rows=n_mach_per_plot,cols=1)
        for j in range(n_mach_per_plot):
            r = random.randrange(0, len(alias_cluster_i))
            fig.add_trace(go.Bar(
                        x=df[df.alias==alias_cluster_i[r]].start_time,
                        y = df[df.alias==alias_cluster_i[r]].per_CPU_use,
                        name = alias_cluster_i[r],
                        marker=dict( line=dict(width=2,
                                    color='DarkSlateGrey'))),
            row=j+1, col=1
            )
        fig.update_layout(title="%s hosts classified in cluster %d using %d classes"%(note,i+1,nclusters))
            
        list_fig.append(fig)

    filename = "results/{0}.html".format(note)
    figures_to_html(list_fig,filename=filename)

def plot_examples_rolling_mean(df_nkmeans,df,n_mach_per_plot,note):
    list_fig = []
    nclusters=len(df_nkmeans.y_pred.unique())
    for i in range(nclusters):
        #separate alias
        alias_cluster_i = df_nkmeans[df_nkmeans.y_pred==i].alias.unique()
        
        #make figure
        fig = make_subplots(rows=n_mach_per_plot,cols=1)
        for j in range(n_mach_per_plot):
            r = random.randrange(0, len(alias_cluster_i))

            alias = alias_cluster_i[r]
            data = (df[df.alias==alias].set_index('start_time')).loc[:,'per_CPU_use']
            data = pd.DataFrame(data)
            sz_win = 20
            short_rolling = data.rolling(window=sz_win).mean()
            # start_date =  short_rolling.index[sz_win-1]
            # end_date = short_rolling.index[-1]

            fig.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling.per_CPU_use,mode= 'lines', 
                                    marker=dict(line=dict(width=2,
                                            color='yellow')), 
                                    name = alias,
                                    marker_color = 'DarkSlateGrey'
                                    ),
                        row=j+1, col=1
                        )
        fig.update_yaxes(range=[0, 80])
        fig.update_layout(title="%s hosts classified in cluster %d using %d classes"%(note,i+1,nclusters))
            
        list_fig.append(fig)

    filename = "results/{0}.html".format(note)
    figures_to_html(list_fig,filename=filename)

def plot_all_machines_rolling_mean(df_nkmeans,df,n_mach_per_plot,note):
    list_fig = []
    nclusters=len(df_nkmeans.y_pred.unique())
    for i in range(nclusters):
        #separate alias
        alias_cluster_i = df_nkmeans[df_nkmeans.y_pred==i].alias.unique()
        
        #make figure
        fig = make_subplots(rows=len(alias_cluster_i),cols=1)
        for r in range(len(alias_cluster_i)):
            alias = alias_cluster_i[r]
            data = (df[df.alias==alias].set_index('start_time')).loc[:,'per_CPU_use']
            data = pd.DataFrame(data)
            sz_win = 20
            short_rolling = data.rolling(window=sz_win).mean()
            # start_date =  short_rolling.index[sz_win-1]
            # end_date = short_rolling.index[-1]

            fig.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling.per_CPU_use,mode= 'lines', 
                                    marker=dict(line=dict(width=2,
                                            color='yellow')), 
                                    name = alias,
                                    marker_color = 'DarkSlateGrey'
                                    ),
                        row=r+1, col=1
                        )
        fig.update_yaxes(range=[0, 80])
        fig.update_layout(height=60*len(alias_cluster_i), width=650,title="%s hosts cluster %d using %d classes"%(note,i+1,nclusters))
        list_fig.append(fig)

    filename = "results/{0}.html".format(note)
    figures_to_html_side_by_side(list_fig,filename=filename)



def save_all_small_single_alias_rolling_mean(df_sv,df_nkmeans,folder_name,note= None):

    '''
        given a df_nkmeans, creates nclusters folders in the specified directory
        containing the images of plots of machines of that cluster
    '''

    scope = PlotlyScope(
        plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
        # plotlyjs="/path/to/local/plotly.js",
    )
    nclusters = len(df_nkmeans.y_pred.unique())
    for c in range(nclusters):

        direc=folder_name+'/cluster_{0}'.format(c)
        Path(direc).mkdir(parents=True, exist_ok=True)

        for i,row in df_nkmeans[df_nkmeans.y_pred==c].iterrows():
            fig = plot_small_single_alias_rolling_mean(df_sv,row['alias'],str(row['Projet'])+' | '+str(row['Role']))
            with open("{0}/figure_{1}.png".format(direc,row['alias']), "wb") as f:
                f.write(scope.transform(fig, format="png"))

def plot_small_single_alias_rolling_mean(df_sv,alias,note= None):

    data = (df_sv[df_sv.alias==alias].set_index('start_time')).loc[:,'per_CPU_use']
    data = pd.DataFrame(data)
    sz_win = 5 #20 before
    short_rolling = data.rolling(window=sz_win).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling['per_CPU_use'],mode= 'lines', 
                            marker=dict(line=dict(width=2,
                                    color='yellow')), 
                            name = alias,
                            marker_color = 'DarkSlateGrey')
                )
    fig.update_yaxes(range=[0, 80])
    if(note):
        fig.update_layout(height=150, width=900, 
                            margin=dict(l=20, r=20, t=27, b=20),
                            title="%s %s"%(alias,note))
    else:
        fig.update_layout(height=150, width=900,
                            margin=dict(l=20, r=20, t=27, b=20),
                            title="%s "%(alias))
    return fig

def save_a_cluster_small_single_alias_rolling_mean(df_sv,df_nkmeans,folder_name,note= None):

    '''
        given a df_nkmeans of a cluster, creates one folder in the specified directory
        containing the images of plots of machines of that cluster
    '''

    scope = PlotlyScope(
        plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
        # plotlyjs="/path/to/local/plotly.js",
    )

    direc='Results/'+folder_name+'/'+note
    Path(direc).mkdir(parents=True, exist_ok=True)

    for i,row in df_nkmeans.iterrows():
        fig = plot_small_single_alias_rolling_mean(df_sv,row['alias'],str(row['Projet'])+' | '+str(row['Role']))
        with open("{0}/figure_{1}.png".format(direc,row['alias']), "wb") as f:
            f.write(scope.transform(fig, format="png"))


def figures_to_html_stacked_and_side_by_side_from_folder(folder_name, filename):
    '''
    plot list of figures in stacked fashion using two columns from files contained in folder_name

    '''
    folder_name = 'Results/'+folder_name
    #get list of files in folder_name
    figs = glob.glob("{0}/*.png".format(folder_name))

    #make two columns of figures
    num_rows = int(math.ceil(len(figs)/2 ))

    direc = 'Results/' +filename
    dashboard = open(direc, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    # discount_rows = 0
    for i in range(2):
        dashboard.write("<div style='display: inline-block; vertical-align: top;'>")
        dashboard.write("<div class='outer'>")
        
        for fig in figs[i*num_rows:num_rows*(i+1)]:
            
            dashboard.write("<div class='inner'>")

            data_uri = base64.b64encode(open(fig, 'rb').read()).decode('utf-8')
            img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            dashboard.write(img_tag)

            dashboard.write("</div>")

        dashboard.write("</div>")
        dashboard.write("</div>")

        # discoount_rows = 1 #next column will possibly have a figure less


    dashboard.write("</body></html>" + "\n")


    
def plot_one_doc_per_cluster(df,df_nkmeans,note=None):

    '''
    create one plot document per cluster displaying the rolling mean of each machine
    light visualizations - Best visu method
    '''
    nclusters = len(df_nkmeans.y_pred.unique())

    #save all individual figure alias
    direc = 'Results/Clustering results-'+str(note)+'/'+ 'Imgs '+str(nclusters)+'kmeans'
    direc_imgs = 'Results/Clustering results-'+str(note)+'/'+ 'Imgs '+str(nclusters)+'kmeans' + '/Each_alias_plot'
    Path(direc_imgs).mkdir(parents=True, exist_ok=True)
    save_all_small_single_alias_rolling_mean(df,df_nkmeans,direc_imgs,note= None)

    #make one .html file per cluster
    for i in range(nclusters):
        figures_to_html_stacked_and_side_by_side_from_folder('{0}/cluster_{1}'.format(direc_imgs,i), 
                                                            '{0}cluster_{1}.html'.format(direc,i))




if __name__ == "__main__":
    cluster1_atemp = ['client14dsy', 'client14xdsy', 'dell3354dsy', 'dell3359dsy',
       'dell3360dsy', 'dell3365dsy', 'dell3366dsy', 'eno24adsy',
       'eno24bdsy', 'eno24cdsy', 'enojarnacdsy', 'enoprj2016dsy',
       'enowip01dsy', 'enowip02dsy', 'enowip03dsy', 'enowip04dsy',
       'enowip05dsy', 'enowip07dsy', 'enowip11dsy', 'enowip12dsy',
       'enowip13dsy', 'enowip14dsy', 'enowip15dsy', 'infra14xdsy',
       'infra15dsy', 'pool08dsy', 'pool09dsy', 'pool10dsy', 'pool11dsy',
       'pool14dsy', 'pool15dsy', 'pool23dsy', 'pool24dsy', 'pool26dsy',
       'pool40dsy', 'pool44dsy', 'pool45dsy', 'pool47dsy', 'pool50dsy',
       'pool51dsy', 'pool53dsy', 'ssdclient15xdsy', 'ssdhwip003dsy',
       'ssdhwip006dsy', 'ssdhwip007dsy', 'ssdhwip009dsy', 'ssdhwip010dsy',
       'ssdhwip011dsy', 'ssdhwip013dsy', 'ssdhwip015dsy', 'ssdhwip016dsy',
       'ssdhwip020dsy', 'ssdhwip021dsy', 'ssdhwip022dsy', 'ssdhwip023dsy',
       'ssdhwip024dsy', 'ssdhwip025dsy', 'ssdinfra15xdsy', 'ssdwip002dsy',
       'ssdwip023dsy', 'ssdwip024dsy', 'wip093dsy', 'wip094dsy',
       'wip095dsy', 'wip096dsy', 'wip097dsy', 'wip102dsy', 'wip103dsy',
       'wip106dsy', 'wip107dsy', 'wip112dsy', 'wip113dsy', 'wip114dsy',
       'wip115dsy', 'wip118dsy', 'wip121dsy', 'wip122dsy', 'wip123dsy',
       'wip124dsy', 'wip126dsy', 'wip128dsy', 'wip131dsy', 'wip133dsy',
       'wip134dsy', 'wip135dsy', 'wip136dsy', 'wip137dsy', 'wip138dsy',
       'wip139dsy', 'wip141dsy', 'wip142dsy', 'wip146dsy', 'wip147dsy',
       'wip148dsy', 'wip150dsy', 'wip152dsy', 'wip153dsy', 'wip154dsy',
       'wip155dsy', 'wip156dsy', 'wip160dsy', 'wip161dsy', 'wip165dsy',
       'wip167dsy', 'wip168dsy', 'wip170dsy', 'wip173dsy', 'wip174dsy',
       'wip176dsy', 'wip179dsy', 'wip181dsy', 'wip182dsy', 'wip183dsy',
       'wip184dsy', 'wip185dsy', 'wip186dsy', 'wip187dsy', 'wip188dsy',
       'wip189dsy', 'wip192dsy', 'wip193dsy', 'wip196dsy', 'wip197dsy',
       'wip198dsy', 'wip199dsy', 'wip200dsy', 'wip203dsy', 'wip204dsy',
       'wip205dsy', 'wip207dsy', 'wip208dsy', 'wip216dsy', 'wip220dsy',
       'wip222dsy', 'wip230dsy', 'wip243dsy', 'wip244dsy', 'wip246dsy',
       'wip247dsy', 'wip248dsy', 'wip249dsy', 'wip250dsy', 'wip251dsy',
       'wip255dsy', 'wip259dsy', 'wip305dsy', 'wip311dsy', 'wip312dsy',
       'wip313dsy', 'wip314dsy', 'wip315dsy', 'wip317dsy', 'wip318dsy',
       'wip319dsy', 'wip320dsy', 'wip322dsy', 'wip323dsy', 'wip328dsy',
       'wip333dsy', 'wip335dsy', 'wip336dsy', 'wip337dsy', 'wip338dsy',
       'wip343dsy', 'wip345dsy', 'wip350dsy', 'wip352dsy', 'wip355dsy',
       'wip356dsy', 'wip357dsy', 'wip358dsy', 'wip359dsy', 'wip363dsy']
    # df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_normalized_data_best_version/merged_df_3kmeans.csv')
    # df_nkmeans_group_of_int = df_nkmeans[df_nkmeans.alias.isin(cluster1_atemp)]
    # df = pd.read_csv('auxiliar/df/df_from10-04on.csv')
    # save_a_cluster_small_single_alias_rolling_mean(df,df_nkmeans_group_of_int,folder_name='Atemporal Clustering',note= 'Cluster1_173_mach')

    figures_to_html_stacked_and_side_by_side_from_folder(folder_name='Atemporal Clustering/Cluster1_173_mach', filename='Atemporal Clustering/atemporal_clustering1.html')