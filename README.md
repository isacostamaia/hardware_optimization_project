# hardware_optimization_project

This project was conceived Time series prediction for hardware optimization

## 1.Context: 
This master thesis project was about optimizing the company’s server park by predicting the CPU occupation level of each one of their machines  (~around 400). This solution used multidimensional time-series data recovered periodically from the park.

## 2.Approach: 
The first stage of this project was focused on data analysis and exploration. Besides visualizing the hardwares temporal record, I studied how CPU data was distributed and could potentially be grouped through clustering of these time-series, using relevant distance criteria as Dynamic Time Warping. Besides that, I explored the possibility of retrieving the frequency of periodical events in each machine’s timeline through Fourier Analysis. After this stage, a research on the state-of the-art methods was conducted and, due to the high non-linearity of data and potentially big importance of long term past data for the task in the matter, I decided to use a Long Short-Term Memory (LSTM) based network for the prediction of novel CPU occupancy level.

## 3.Particularities of the database: 
Since the high CPU usage events ran in each machine were sporadic, the CPU records were mainly representing the idle state, which implied a database that was high-occupancy-level sparse. Therefore the prediction of those levels of activity had low accuracy. The most traditional techniques of balancing databases for improving prediction capability of misrepresented classes is often based on the exclusion or insertion of crafted data points, which can easily alter the phenomena represented on it, specially in the cases of time series. 

## 4.Proposed Solution: 
Instead of artificially changing the database to solve the unbalancing problem, I studied the relevance of importing a solution from recent breakthroughs on Natural Language Process (NLP) framework: the attention mechanism. Its concept is based on a second neural network that works as a decoder of the information encoded by the LSTM network, which is the hidden states sequence. This second network learns from data the appropriate weights for the encoder’s hidden states giving more importance to certain time-steps of the input sequences. In later research, I found recent work that explored the benefit of different attention mechanism implementations on time-series prediction tasks, which somehow confirmed the pertinence of applying it to this project.

## 5.Architecture: 
The network’s resulting architecture can be seen in below:
![lstm att net](https://user-images.githubusercontent.com/23039791/193483748-ec74fca2-2e98-4b50-a46e-3b54395492ba.jpg)


## 6.Results: 
The addition of the attention mechanism provoked an improvement on predicting high-CPU-usage classes compared to the original time series, indicating its potential in improving the network accuracy on further exploration.
