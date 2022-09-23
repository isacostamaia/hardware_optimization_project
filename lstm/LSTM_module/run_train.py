from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.csv_logs import CSVLogger

from .lstm_modularized import LSTMRegressor, DataModule

from . import settings


def run_train():

    seed_everything(1)

    csv_logger = CSVLogger('./', name='lstm', version= settings.VERSION),


    #not used for the moment
    early_stop_callback = EarlyStopping(
    monitor='validation_loss_batch',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='max'
    )

    if settings.GPU_DESIRED: 
        gpu=-1
    else:
        gpu=0

    trainer = Trainer(
        max_epochs= settings.PARAM['max_epochs'],
        logger=csv_logger,
        gpus=gpu,
    #     row_log_interval=1,
        progress_bar_refresh_rate=2,
    #     callbacks=[early_stop_callback],
    )

    dm = DataModule(
        seq_len = settings.PARAM['seq_len'],
        batch_size = settings.PARAM['batch_size'], 
        # num_workers= 5,
        pin_memory=True
    )
    
    model = LSTMRegressor(
        n_features = len(dm.hostnames),
        hidden_size = settings.PARAM['hidden_size'],
        seq_len = settings.PARAM['seq_len'],
        batch_size = settings.PARAM['batch_size'],
        criterion = settings.PARAM['criterion'],
        num_layers = settings.PARAM['num_layers'],
        dropout = settings.PARAM['dropout'],
        learning_rate = settings.PARAM['learning_rate'],
        epochs = settings.PARAM['max_epochs'],
        start_date = settings.START_DATE,
        end_date = settings.END_DATE,
        num_weeks = settings.NUM_WEEKS
    )


    trainer.fit(model, dm)

    return model, dm

     # Run as program
if __name__ == '__main__':
    run_train()