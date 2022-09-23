import random

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


            
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