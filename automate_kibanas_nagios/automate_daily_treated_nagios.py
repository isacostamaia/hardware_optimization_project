import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers

import df_sv, df_alias_all



def put_in_elastic(df_alias_all):
    print("Sending to elasticsearch...")
    es = Elasticsearch(['http://wip286dsy:9100',])
    
    # Create id based on host name and date
    ids = [str(date)+alias for date,alias in zip(df_alias_all.index,df_alias_all.alias)] 
    
    def gendata():
        count=0
        for index, row in df_alias_all.reset_index().iterrows():
            yield {
                "_index": "nagios_group_data", "_type":"df", "_id":ids[count],

                "start_time": row['start_time'],
                "avg_cpu_use": row['avg_cpu_use'],
                "max_perc_cpu": row['max_perc_cpu'],
                "min_perc_cpu": row['min_perc_cpu'],
                "std_dev_cpu": row['std_dev_cpu'],
                "num_servicechecks": row['num_servicechecks'],
                "alias": row['alias'],
                "group": row['group'],
                "level": row['level'],
                "os": row['os'],
            }
            count+=1
    try:
        # make the bulk call, and get a response
        response = helpers.bulk(es, gendata())

        #response = helpers.bulk(elastic, actions, index='employees', doc_type='people')
        print ("\nRESPONSE:", response)
    except Exception as e:
        print("\nERROR:", e)
    

if __name__ == "__main__":
    df_alias_all = df_alias_all.make_df_all_alias(df_sv_from_db=True, mode='daily')
    put_in_elastic(df_alias_all)