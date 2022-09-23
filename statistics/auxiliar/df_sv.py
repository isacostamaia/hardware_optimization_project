import psycopg2
import pymysql
import pandas as pd
from sqlalchemy import create_engine

def get_treated_df_sv():

    ###GET

    # Create an engine instance
    #dialect+driver://username:password@host:port/database
    alchemyEngine   = create_engine('mysql+pymysql://ndoutils:nagiosadmin@lin033dsy/nagios', pool_recycle=3600);

    # Connect to PostgreSQL server
    dbConnection    = alchemyEngine.connect();

    query_servchecks_only_ess = "select servicecheck_id,service_object_id,start_time,end_time,execution_time,output,convert(substring(perfdata, 7), unsigned integer) per_CPU_use from nagios.nagios_servicechecks where output = 'hw_usage_cpu' order by servicecheck_id desc; "

    query_serv_only_ess = "SELECT service_id,service_object_id,host_object_id,display_name FROM nagios.nagios_services;"

    query_hosts_only_ess = "SELECT host_id,host_object_id,alias,icon_image_alt os,display_name FROM nagios.nagios_hosts;"

    df_servicechecks = pd.read_sql(query_servchecks_only_ess, dbConnection)
    df_services =  pd.read_sql(query_serv_only_ess, dbConnection)
    df_hosts = pd.read_sql(query_hosts_only_ess, dbConnection)

    pd.set_option('display.expand_frame_repr', False)

    # Close the database connection
    dbConnection.close()

    ###TREAT
    df_servicechecks.start_time = df_servicechecks.start_time.dt.tz_localize('utc')
    df_servicechecks.end_time = df_servicechecks.end_time.dt.tz_localize('utc')

    # merge
    df_sv = df_servicechecks.merge(df_services,how='inner')
    df_sv = df_sv.merge(df_hosts,how='inner', on='host_object_id',suffixes=('_service','_host'))

    #make all alias in lower case and adjust os name
    df_sv.alias=df_sv.alias.str.lower() 
    df_sv.os.loc[df_sv.os=='Windows Server']='windows'
    df_sv.os.loc[df_sv.os=='Linux']='linux'

    #drop nan
    df_sv.dropna(subset=['start_time','per_CPU_use'], how='any', inplace=True)

    return df_sv

def save_df_sv(df_sv):
    df_sv.to_csv(r'df/df_sv.csv')

# def main():
#     print('q')
#     save_df_sv(get_treated_df_sv())
#     print('bele')

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
#    main()
    save_df_sv(get_treated_df_sv())

# main()