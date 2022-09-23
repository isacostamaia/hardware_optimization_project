
import numpy as np
import pandas as pd

from df_sv import get_treated_df_sv

#Make one df per alias
def retrieve_df_alias(df_sv,alias,os,level="other",group="other"):
    
    df_alias = df_sv[df_sv.alias==alias].set_index('start_time')

    #make auxiliar df and add column to take max,min,std of host in a certain hour
    df_alias_aux = df_alias.copy() 
    df_alias_aux['max_perc_cpu'] = df_alias.per_CPU_use
    df_alias_aux['min_perc_cpu'] = df_alias.per_CPU_use
    df_alias_aux['std_dev_cpu'] = df_alias.per_CPU_use


    #ignore 0 values to compute min of cpu
    zero_to_nan = lambda x: np.nan if x==0 else x
    df_alias_aux['min_perc_cpu'] = [zero_to_nan(val) for val in df_alias_aux.per_CPU_use]

    #before aggregating auxiliar df: add column to say if host has servicechecks in each time interval
    df_alias_aux['num_servicechecks'] = [1]*len(df_alias_aux)

    #aggregate the auxiliar df directly by hour
    df_alias_aux = df_alias_aux.resample('1H').agg({'max_perc_cpu':np.nanmax,
                                                    'min_perc_cpu':np.nanmin,
                                                    'std_dev_cpu':np.nanstd,
                                                    'num_servicechecks':np.sum})

    #add column to say if host has a service check in a certain hour
    df_alias_aux["has_sv"] = [0 if num_sv==0 else 1 for num_sv in df_alias_aux.num_servicechecks]
    
    #add column with os name
    df_alias_aux["os"] = [os]*len(df_alias_aux)

    #add columns with host name
    df_alias_aux["alias"] = [alias]*len(df_alias_aux)

    #add column with level name 
    df_alias_aux["level"] = [level]*len(df_alias_aux)

    #add column with group name
    df_alias_aux["group"] = [group]*len(df_alias_aux)

    #now aggregate main df by minute
    df_alias_sum = df_alias.resample('30S').agg({'per_CPU_use': np.sum})

    #then aggregate main df by average of cpu use in one hour
    df_alias_sum = df_alias.resample('1H').agg({'per_CPU_use': np.mean})

    #rename its column
    df_alias_sum.rename(columns={"per_CPU_use":'avg_cpu_use'},inplace=True)

    #in this main df, transforme per_CPU_use in Nan if it doesn't have a servicecheck in that hour
    df_alias_sum['avg_cpu_use'] = [np.nan if has_sv==0 else per_cpu for has_sv,per_cpu in 
                                   zip(df_alias_aux.has_sv,df_alias_sum.avg_cpu_use)]


    #aggregate info in a single df
    df_alias = df_alias_aux.copy()
    df_alias['avg_cpu_use'] = df_alias_sum.avg_cpu_use
    return(df_alias)

#retrieve hourly average and statistics from each host
def make_df_all_alias(df_sv_from_db=False):
       
    pd.options.mode.chained_assignment = None  # default='warn'

    if(df_sv_from_db==True):
        df_sv = get_treated_df_sv()
    else:
        df_sv = pd.read_csv('df_sv.csv')

    ############################ Make one df per alias

    release_machines = pd.read_json('release_machines.json')

    oss=['windows', 'linux']
    groups = ['preint_build', 'prj_build', 'replay']


    list_df_alias = []
    for os in oss:
        #Machines of the same os that are in Nagios but are not in JSON file
        rest_alias = np.setdiff1d(df_sv[df_sv.icon_image_alt==os].alias.unique(),release_machines.machine.unique())
        for group in groups:

            release_group_os =  release_machines[(release_machines.role==group) & (release_machines.os==os)]
            levels = release_group_os.level.unique()
            alias_group_os = [release_group_os[release_group_os.level==l].machine.values for l in levels]
            alias_group_os_flat =release_group_os.machine.unique()

            for i, all_alias_one_level in enumerate(alias_group_os):
                for alias in all_alias_one_level:
                    list_df_alias.append(retrieve_df_alias(df_sv,alias,os,levels[i],group))

        for other_alias in rest_alias:

            list_df_alias.append(retrieve_df_alias(df_sv,other_alias,os))
            
   ################################
    df_alias_all = pd.concat(list_df_alias)


    #turn into float values
    df_alias_all['avg_cpu_use'] = df_alias_all.avg_cpu_use.astype(float)
    df_alias_all['max_perc_cpu'] = df_alias_all.max_perc_cpu.astype(float)
    df_alias_all['min_perc_cpu'] = df_alias_all.min_perc_cpu.astype(float)
    df_alias_all['std_dev_cpu'] = df_alias_all.std_dev_cpu.astype(float)
    df_alias_all['has_sv'] = df_alias_all.has_sv.astype(float)
    
    #replace nan by None
    df_alias_all = df_alias_all.where(pd.notnull(df_alias_all), None)
    return df_alias_all

def save_df_alias_all(df_alias_all):
    df_alias_all.to_csv(r'df/df_alias_all.csv')

def save_machines_identity():
    df_alias_all = pd.read_csv('df_alias_all.csv')
    iden = df_alias_all.groupby('alias').agg({'os':lambda x: x.unique(),'level': lambda x: x.unique(), 'group': lambda x: x.unique()})
    iden.to_csv(r'df/df_machines_identity.csv')

def retrieve_machines_identity():
    return pd.read_csv('df/df_machines_identity.csv')

# def main():


# main()    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
#    main()
    print('boooo')
    save_machines_identity()
    a = retrieve_machines_identity()
    print('ok')
