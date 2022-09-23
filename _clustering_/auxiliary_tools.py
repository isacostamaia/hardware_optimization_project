import math
from datetime import datetime, timedelta

import base64
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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

def plot_small_single_alias(df_sv,alias,note= None): #plots a small figure from a given alias 
    
    # #plot only data from a week ago
    # df_sv = df_sv[(df_sv.start_time>=str(datetime.now()-timedelta(weeks=1)))]
        
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


def plot_small_single_alias_rolling_mean(df_sv,alias,axis_range_multiplier=1,note= None):

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
    y_lim = df = 80*axis_range_multiplier
    fig.update_yaxes(range=[0, y_lim])
    if(note):
        fig.update_layout(height=150, width=900, 
                            margin=dict(l=20, r=20, t=27, b=20),
                            title="%s %s"%(alias,note))
    else:
        fig.update_layout(height=150, width=900,
                            margin=dict(l=20, r=20, t=27, b=20),
                            title="%s "%(alias))
    return fig

def save_all_small_single_alias_rolling_mean(df_sv,df_nkmeans,folder_name,axis_range_multiplier=1,note= None):

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

        direc=folder_name+'/cluster_{0}'.format(c+1)
        Path(direc).mkdir(parents=True, exist_ok=True)

        for i,row in df_nkmeans[df_nkmeans.y_pred==c].iterrows():
            fig = plot_small_single_alias_rolling_mean(df_sv,row['alias'],axis_range_multiplier,str(row['Projet'])+' | '+str(row['Role']))
            with open("{0}/figure_{1}.png".format(direc,row['alias']), "wb") as f:
                f.write(scope.transform(fig, format="png"))




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

def plot_list_machines(df_sv,list_alias,name_folder): #generate folder with a file for each machine in the list (name of file being the name of alias)
    #name_folder of type 'Machines_Folder'

    #plot only data from a week ago
    df_sv = df_sv[(df_sv.start_time>=str(datetime.now()-timedelta(weeks=1)))]

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

def plot_all_machines_rolling_mean(df_nkmeans,df,note):
    list_fig = []
    nclusters=len(df_nkmeans.y_pred.unique())
    for i in range(nclusters):
        #separate alias
        alias_cluster_i = df_nkmeans[df_nkmeans.y_pred==i].alias.unique()
        
        #make figure
        fig = make_subplots(rows=len(alias_cluster_i),cols=1)
        for r in range(len(alias_cluster_i)):
            alias = alias_cluster_i[r]
            #plot only data from a week ago
            data = (df[(df.alias==alias)&(df.start_time>=str(datetime.now()-timedelta(weeks=1)))].set_index('start_time')).loc[:,'per_CPU_use']
            data = pd.DataFrame(data)
            sz_win = 5
            short_rolling = data.rolling(window=sz_win).mean()
            # start_date =  short_rolling.index[sz_win-1]
            # end_date = short_rolling.index[-1]

            fig.add_trace(go.Scatter(x = short_rolling.index,y=short_rolling.per_CPU_use,mode= 'lines', 
                                    marker=dict(line=dict(width=2,
                                            color='yellow')), 
                                    name = alias,
                                    marker_color = 'DarkSlateGrey',
                                    ),
                        row=r+1, col=1
                        )

        fig.update_yaxes(range=[0, 80])
        fig.update_layout(height=70*len(alias_cluster_i), width=650,title="%s hosts cluster %d using %d classes"%(note,i+1,nclusters))
        list_fig.append(fig)

    filename = "results/{0}.html".format(note)
    figures_to_html_side_by_side(list_fig,filename=filename)

def figures_to_html_stacked_and_side_by_side(figs, filename):
    #plot list of figures in stacked fashion using two columns

    #make two columns of figures
    num_rows = int(math.ceil(len(figs)/2 ))

    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    dashboard.write("<div style='white-space:nowrap; vertical-align: top;'>")
    for i in range(2):
        dashboard.write("<div style='display: inline-block; vertical-align: top;'>")
        dashboard.write("<div class='outer'>")


        for fig in figs[i*num_rows:num_rows*(i+1)]:
            dashboard.write("<div class='inner'>")
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
            dashboard.write("</div>")

        dashboard.write("</div>")
        dashboard.write("</div>")

        num_rows = int(len(figs)/2) #next column will possibly have a figure less
    dashboard.write("</div>")

    dashboard.write("</body></html>" + "\n")

def figures_to_html_stacked_and_side_by_side_from_folder(folder_name, filename):
    '''
    plot list of figures in stacked fashion using two columns from files contained in folder_name

    '''
    #get list of files in folder_name
    figs = glob.glob("{0}/*.png".format(folder_name))

    #make two columns of figures
    num_rows = int(math.ceil(len(figs)/2 ))

    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")

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

        num_rows = int(len(figs)/2) #next column will possibly have a figure less


    dashboard.write("</body></html>" + "\n")

if __name__ == "__main__":
    # df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_normalized_data_best_version/merged_df_3kmeans.csv')
    # df = pd.read_csv('df/df_from10-04on.csv')
    # save_all_small_single_alias_rolling_mean(df,df_nkmeans,'Folder_figures')
    # b=[]
    # pathlist = Path('Folder_figures/cluster_0').glob('**/*.png')
    # for path in pathlist:
    #     # because path is object not string
    #     path_in_str = str(path)
    #     print(path_in_str)
    #     b.append(path_in_str)

    # print(list(pathlist))
    
    print(glob.glob("Folder_figures/cluster_0/*.png"))

    # figures_to_html_stacked_and_side_by_side_from_folder('Folder_figures/cluster_0', 'visu_cluster1.html')
    figures_to_html_stacked_and_side_by_side_from_folder('Folder_figures/cluster_1', 'visu_cluster2.html')
    figures_to_html_stacked_and_side_by_side_from_folder('Folder_figures/cluster_2', 'visu_cluster3.html')