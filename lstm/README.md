# Predicting Machines 

## Requirements

* [Python 3.7](https://www.python.org/)
* [virtualenv](https://pypi.org/project/virtualenv/)

## Install

Create and activate a virtualenv
```
virtualenv --python=[path_to_python3.7].exe env
env\Scripts\activate
```

Install libraries
```
pip install -r requirements.txt
```

## Usage
### Regular LSTM Network
in "./" run
```
python -m LSTM_module.train_and_predict 
```

### Attention Based LSTM network (encoder-decoder)
in "./" run
```
python "Notebooks and jupytetxts/simple_att_pack/script_att.py" 
```

#### Above models results 
in "./" run
```
python -m jupyter notebook 
```
Open "Notebooks and jupytetxts/Evaluate Models Loop.py" and analyse the results written in "./lstm/My_loop_global_var_and_metrics.txt".
### LSTM Network for Ordinal Classification
in "./" run
```
python -m jupyter notebook  
```
Open "/Notebooks and jupytetxts/LSTM-Encoding.py" notebook.

### Previously trained LSTMs
in "./" run
```
python -m jupyter notebook  
```
Open 
*  "/Notebooks and jupytetxts/Category predictions from already trained regression model - 2051.ipynb" 
*  "/Notebooks and jupytetxts/Category predictions from already trained regression model - 7395.ipynb" 


## Details 
**LSTM Network**\
Network composed by *n* LSTM layers (*n* and other hyperparameters are user-defined in LSTM_module/settings.py) followed by fully connected layer. Predicts one hour ahead for all the machines. \
Examples of Results:
* *\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\Final\2051_loop2_ssdwip016dsy*
* *\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\Final\7395_loop2_ssdwip016dsy*


**Attention Based LSTM network**\
Variant of the network described by Bahdanau et. al and explained in [this article](https://blog.floydhub.com/attention-mechanism/). Architecture scheme is illustrated in *\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\Final\attnet.PNG* (in file explorer).
Predicts one hour ahead for all the machines. \
Example of Result:

* *\\ariel1dsy\home\ENOVIA\IPLM-WIP\97-Release\DevOps\HWUsage\Final\Attention\attention_same_hidden_teacher_forcing* 
