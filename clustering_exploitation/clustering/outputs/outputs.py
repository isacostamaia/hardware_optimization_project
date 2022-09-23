"""
Write clustering results to output files
"""
import os
import io
import base64

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

_DPI = 96
_, _axix = plt.subplots(figsize=(500/_DPI, 170/_DPI))

def get_machine_figure(dataset, max_y):
    """
    Return a base64 figure image for given machine dataset
    dataset: dataset containing (date, cpu) series
    max_y: maximum y scale value
    """
    _axix.plot(dataset.date, dataset.cpu)
    plt.minorticks_off()
    _axix.xaxis.set_tick_params(labelsize=5.5, rotation=55, which='major')
    _axix.xaxis.set_major_locator(plt.MaxNLocator(20))
    _axix.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d_%H-%M'))
    for label in _axix.xaxis.get_majorticklabels():
        label.set_horizontalalignment('right')
    _axix.yaxis.set_major_locator(plt.MaxNLocator(6))
    _axix.yaxis.set_tick_params(labelsize=5.5, which='major')
    _axix.axes.set_ylim(0, max_y)

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='png')
    plt.cla()
    io_bytes.seek(0)
    base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
    io_bytes.close()
    return base64_image

def generate_cluster_html(hosts_clusters, data_df, max_y, label, output_directory):
    """
    Generate an html file for each input cluster
    hosts_clusters: hostname to cluster index mapping
    data_df: per host cpu time series
    max_y: maximum y scale value
    label: file naming label
    output_directory: path to write html files to
    """
    # Get machine info from "Machines_suivi" file
    machine_suivi_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources',
        'Machines_Suivi.csv'
    )
    machines_df = pd.read_csv(machine_suivi_path, sep=';')
    machines_df.loc[:, 'Name'] = [x.lower() for x in machines_df.Name]
    machines_df = machines_df[['Name', 'Projet', 'Role']]
    hosts_clusters = hosts_clusters.copy().merge(
        machines_df,
        how='left',
        left_on='hostname',
        right_on='Name'
    )
    clusters = hosts_clusters.cluster.unique()
    for cluster in clusters:
        html = '<div style="display: flex;flex-wrap: wrap;font-family: calibri">'

        for _, item in hosts_clusters[hosts_clusters.cluster == cluster].iterrows():
            figure_b64 = get_machine_figure(
                dataset=data_df[data_df.hostname == item.hostname],
                max_y=max_y
            )
            html += (
                '<div>' +
                    '<div>' +
                        '{} - {} | {}'.format(item.hostname, item.Projet, item.Role) +
                    '</div>' +
                    '<img src="data:image/png;base64,{0}">'.format(figure_b64) +
                '</div>'
            )

        html += '</div>'
        html_path = os.path.join(
            output_directory,
            'cluster_{}_{}.html'.format(label, cluster)
        )
        with open(html_path, 'w', errors='replace') as html_file:
            html_file.write(html)

def write_cluster_machine_lists(hosts_clusters, output_directory):
    """
    Write one txt file per cluster containing its list of machines
    """
    clusters = hosts_clusters.cluster.unique()
    for cluster in clusters:
        machines = ''
        for _, item in hosts_clusters[hosts_clusters.cluster == cluster].iterrows():
            machines += item.hostname + '\n'
        cluster_path = os.path.join(output_directory, 'cluster_{}.txt'.format(cluster))
        with open(cluster_path, 'w') as cluster_file:
            cluster_file.write(machines)

def write_clustering_results(initial_df, normalized_df, tskm_result, output_directory):
    """
    Write clustering result files to the specified directory
    initial_df: dataframe with initial dataset
    normalized_df: dataframe with normalized dataset (clustering input)
    tskm_result: TimeSeriesKMeans result object
    output_directory: path to results directory
    """
    # Write tskm json file
    tskm_result.to_json(os.path.join(output_directory, 'tskm.json'))

    hostnames = initial_df.hostname.unique()
    hosts_clusters = pd.DataFrame(
        {'hostname': hostnames, 'cluster': tskm_result.labels_}
    ).sort_values('cluster')

    # Write clusters machines lists
    write_cluster_machine_lists(hosts_clusters, output_directory)

    # Put figures in html files per cluster
    generate_cluster_html(
        hosts_clusters=hosts_clusters,
        data_df=initial_df,
        max_y=100,
        label='initial',
        output_directory=output_directory
    )

    max_y = normalized_df.cpu.max()
    generate_cluster_html(
        hosts_clusters=hosts_clusters,
        data_df=normalized_df,
        max_y=max_y,
        label='normalized',
        output_directory=output_directory
    )
