
from retrieve_and_prepare_data import db_to_df
from Regression_many_params import prediction_one_machine_many_params
import settings

def predict_many_machines():
    df_all = db_to_df()
    
    for alias in df_all.hostname.unique():
        df = df_all[df_all.hostname==alias]
        
        prediction_one_machine_many_params(df, len(df_all.hostname.unique()))


if __name__ == '__main__':
    # for wip329dsy
    predict_many_machines()

