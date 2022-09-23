## Secondary branches
*  modified_attention_pack\
Has different "Notebooks and Jupytexts/simple_att_pack/" folder version. Tests an implementation of Attention encoder-decoder network where hidden state in the decoder is different for each one of the machines. Each one of them is issued from a convolutional network that summarizes the information contained in the timeline of the correspondant machine.

* modified_att_pack_hidden_lstm\
Same as the above but each of the hidden states is originated from a third LSTM network. Code has bug (one tensor should be detached from computing graph). Still has to test the effect.

* brolle_intro_noise\
Tested the effect of introducing prediction noise during training.  Inconclusive effect.

## Useful links
* [Encoder-Decoder Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)
* [Ordinal Classification](https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99)
* [Rank consistent classification (CORAL framework)](https://github.com/Raschka-research-group/coral-cnn/)
* [Temporal pattern attention for multivariate time series forecasting](https://link.springer.com/content/pdf/10.1007/s10994-019-05815-0.pdf)
* [AT-LSTM: An Attention-based LSTM Model for Financial Time Series Prediction](https://iopscience.iop.org/article/10.1088/1757-899X/569/5/052037/pdf)
