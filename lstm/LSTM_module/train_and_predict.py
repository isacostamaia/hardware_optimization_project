# from .LSTM_module import settings
import base64
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO

from ruamel.yaml import YAML
import yaml
import torch.nn as nn
from soft_dtw_cuda import SoftDTW
from pytorch_forecasting.metrics import MAPE

from .run_train import run_train
from .run_predictions import get_predictions
from .output_images import generate_predictions_html, get_loss_img, compute_test_error
from . import settings



def train_and_predict():
    '''
        Train, predict and save output figures (predictions and loss)
    '''


    _da_criterion = nn.MSELoss()#just initial value, or value if we do not run in loop
    settings.PARAM['criterion'] = _da_criterion
    _da_batch_size = 1  #just initial value, or value if we do not run in loop
    settings.PARAM['batch_size'] = _da_batch_size
    _da_seq_len = 4
    settings.PARAM['seq_len'] = _da_seq_len
    _da_learning_rate = 0.0005
    settings.PARAM['learning_rate'] = _da_learning_rate  
    _da_epochs = 1
    settings.PARAM['max_epochs'] = _da_epochs
    _da_num_layers = 2
    settings.PARAM['num_layers'] = _da_num_layers
    _da_num_weeks = 9
    settings.NUM_WEEKS = _da_num_weeks
    # _da_date = datetime.fromisoformat('2021-03-25 11:00:00') - timedelta(weeks = 9)
    # settings.DB_QUERY_ARGS['start_date'] = _da_date
    # settings.DB_QUERY_ARGS['end_date'] = None
    _da_version = 'trash'
    settings.VERSION = _da_version
    _da_hidden_size = 200
    settings.PARAM['hidden_size'] = _da_hidden_size
    _da_dropout = 0.2
    settings.PARAM['dropout'] = _da_dropout

    global_direc = './lstm/'
    direc = global_direc + '{}/'.format(settings.VERSION)

    #train
    start_train = datetime.now()
    model, dm = run_train()
    end_train = datetime.now()
    duration_train = {'duration_train_min': ((end_train - start_train).total_seconds())/60}
        #add duration_train to yaml file
    yaml = YAML()
    yaml.preserve_quotes = True

    with open('{}hparams.yaml'.format(direc)) as f:
        params_dic = yaml.load(f)

    params_dic.update(duration_train)

    with open('{}hparams.yaml'.format(direc), 'w') as yaml_file:
        yaml.dump(params_dic, yaml_file)

    #predict
    preds_df = get_predictions(model, dm)
    
    #predictions error: error = dist_dtw_, mixed_err_
    dtw_, mix_ = compute_test_error(dm.dataset, preds_df, len(dm.train_all))
    #prediction html
    generate_predictions_html([dtw_, mix_], dm.dataset, preds_df, len(dm.train_all))
    print("predictions images saved")

    #save train and val loss
    fig64 = get_loss_img()
    im = Image.open(BytesIO(base64.b64decode(fig64)))
    im.save('{0}train_val_loss.png'.format(direc), 'PNG')
    print("loss images saved")

    #save predictions metadata
        #write criterion in more readable manner
    if isinstance(settings.PARAM['criterion'], nn.MSELoss):
        crit = "MSE"
    elif isinstance(settings.PARAM['criterion'], SoftDTW):
        crit = "SoftDtw_gamma={}".format(settings.PARAM['criterion'].gamma)
    elif isinstance(settings.PARAM['criterion'], MAPE): 
        crit = "MAPE" 
        #number of weeks
    num_weeks = settings.NUM_WEEKS

    for h, dtw, mix in zip(dm.dataset.columns, dtw_, mix_):
        with open('{}.txt'.format('./lstm/' + settings.LOOP_NAME + "_global_var_and_metrics"),"a+") as f:
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    settings.VERSION,
                    h,
                    num_weeks,
                    settings.DB_QUERY_ARGS['start_date'],
                    crit,
                    settings.PARAM['seq_len'],
                    settings.PARAM['learning_rate'],
                    settings.PARAM['max_epochs'], 
                    settings.PARAM['num_layers'],                   
                    settings.PARAM['batch_size'],
                    settings.PARAM['hidden_size'],
                    settings.PARAM['dropout'],
                    duration_train['duration_train_min'],
                    dtw,
                    mix
            ))
    print("metadata saved")


# Run as program
if __name__ == '__main__':
    train_and_predict()
