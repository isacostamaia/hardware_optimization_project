import os

from Regression import prediction_one_machine
from output import get_loss_plot, get_predict_plot, generate_machine_html
from retrieve_and_prepare_data import db_to_df
import settings

def change_parameters(hidden_dim = settings.HIDDEN_SIZE,
                    num_layers = settings.NUM_LAYERS,
                    train_window = settings.TRAIN_WINDOW,
                    epochs = settings.EPOCHS,
                    lr = settings.LR,
                    use_filtered_data = settings.USE_FILTERED_DATA
                    ):

    settings.HIDDEN_SIZE = hidden_dim
    settings.NUM_LAYERS = num_layers
    settings.TRAIN_WINDOW = train_window
    settings.EPOCHS = epochs
    settings.LR = lr
    settings.USE_FILTERED_DATA = use_filtered_data

# settings.NAME = name
# settings.INPUT_SIZE = 1
# settings.OUTPUT_SIZE = 1

def get_string_parameters():
    config = '''
                name = {6} <br>
                epochs = {3} <br>
                lr = {4} <br>
                train_window = {2} <br>
                hidden_dim = {0} <br>
                num_layers = {1} <br>
                use_filtered_data = {5} <br>
                input_size = {7} <br>
                output_size = {8} <br>
            '''. format(
                        settings.HIDDEN_SIZE, 
                        settings.NUM_LAYERS,
                        settings.TRAIN_WINDOW,
                        settings.EPOCHS,
                        settings.LR,
                        settings.USE_FILTERED_DATA,
                        settings.NAME,
                        settings.INPUT_SIZE,
                        settings.OUTPUT_SIZE,
                 )
    return config

def prediction_one_machine_many_params(df,num_mach):

    #update name and directory
    settings.NAME = '{0}_{1}_{2}'.format(df.iloc[0].Projet, df.iloc[0].Role, df.iloc[0].hostname)
    settings.MACHINE_DIRECTORY = settings.DIRECTORY + settings.NAME

    list_loss_plot = []
    list_pred_plot = []
    list_config = []

    # epochs_ = [5,50,300,500]
    # num_layers_ = [1,2,5]
    # hidden_dim_ = [100,200,500]
    # train_window_ = [5,25,75]
    # lr_ = [0.00005,0.0005,0.001,0.5]
    # use_filtered_data_ = [False, True]

    # epochs_ = [5]
    # num_layers_ = [1]
    # hidden_dim_ = [100]
    # train_window_ = [5]
    # lr_ = [0.0005]
    # use_filtered_data_ = [False]
    
    epochs_ = [5,200,500]
    num_layers_ = [1,2]
    hidden_dim_ = [100,250]
    train_window_ = [5,60]
    lr_ = [0.00005,0.0005,0.05]
    use_filtered_data_ = [False]

    total = len(epochs_)*len(num_layers_)*len(hidden_dim_)*len(train_window_)*len(lr_)*len(use_filtered_data_)*num_mach

    i = 1

    for use_filtered_data in use_filtered_data_:
        for num_layers in num_layers_:
            for hidden_dim in hidden_dim_:
                for train_window in train_window_:
                    for lr in lr_:
                        for epochs in epochs_:

                            change_parameters(epochs = epochs,
                                                num_layers=num_layers,
                                                lr=lr, 
                                                train_window=train_window, 
                                                hidden_dim=hidden_dim, 
                                                use_filtered_data=use_filtered_data)

                            print("Iteration {0}/{1}".format(i,total))
                            config = get_string_parameters()
                            print(config)
                            loss_plot, pred_plot = prediction_one_machine(df)

                            list_loss_plot.append(loss_plot)
                            list_pred_plot.append(pred_plot)
                            list_config.append(config)
                            i += 1
    generate_machine_html(list_loss_plot, list_pred_plot, list_config)

# Run as program
if __name__ == '__main__':
    # for wip329dsy
    df = db_to_df()
    all_alias = df.hostname.unique()
    prediction_one_machine_many_params(df[df.hostname==all_alias[0]])
