import pandas as pd
from elasticsearch import Elasticsearch, helpers

import df_sv

def put_in_elastic(df_sv):
    print("Sending to elasticsearch...")
    es = Elasticsearch(['http://wip286dsy:9100',])
    
    def gendata():
        count=0
        for index, row in df_sv.reset_index().iterrows():
            yield {
                "_index": "raw_nagios", "_type":"df", "_id":row.servicecheck_id,

                "start_time": row['start_time'],
                "avg_cpu_use": row['per_CPU_use'],
                "alias": row['alias'],
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
    # df_sv = df_sv.get_treated_df_sv(daily=True)
    df_sv = df_sv.get_treated_df_sv(mode='hourly')
    put_in_elastic(df_sv)


