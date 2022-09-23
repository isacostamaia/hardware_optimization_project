# TPA-LSTM

Original Implementation of [''Temporal Pattern Attention for Multivariate Time Series Forecasting''](https://arxiv.org/abs/1809.04206).

## Dependencies

* python3.6

You can check and install other dependencies in `requirements.txt`.

```
$ pip install -r requirements.txt
# to install TensorFlow, use the below:


$pip install [path to file below]

\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\TPA-LSTM\1.7.1\py36\CPU\avx2\tensorflow-1.7.1-cp36-cp36m-win_amd64.whl

```

## Dataset

The full dataset that should be downloaded can be found at 

```
https://github.com/laiguokun/multivariate-time-series-data/blob/master/solar-energy
```

A minimalistic version is present in 
```
\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\TPA-LSTM\solar_AL3
```


## Usage

The following example usage shows how to train and test a TPA-LSTM model on MuseData with settings used in this work.

### Training

```
$ python main.py --mode train --attention_len 16 --batch_size 32  --data_set solar_AL --highway 16 --mts 5 --dropout 0.2 --learning_rate 1e-5 --model_dir ./models/model --num_epochs 40 --num_layers 3 --num_units 338

```

### Testing

```
$ python main.py --mode test \
    --attention_len 16 \
    --batch_size 32 \
    --data_set muse \
    --dropout 0.2 \
    --learning_rate 1e-5 \
    --model_dir ./models/model \
    --num_epochs 40 \
    --num_layers 3 \
    --num_units 338
```
