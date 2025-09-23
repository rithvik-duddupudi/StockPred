**StockPred**, a project that gathers machine learning and deep learning models for stock forecasting.

## Models
I coded an LSTM RNN and a Simple Signal Rolling Agent to perform stock forecasting on various stocks.

## Frontend
I then coded the frontend for the LSTM RNN and Simple Signal Rolling Agent inside Tensorflow JS, and you can try it at [the StockPredict website](https://stockpredict999.netlify.app/). In the application, you can upload an historical CSV 

## Results

### Agent Results

**This agent is only able to buy or sell 1 unit per transaction.**

Signal rolling agent

<img src="agents-results/signal-rolling-agent.png" width="70%" align="">


### Model Results

For train-test split: 

1. Train dataset was derived from the starting timestamp until the last 30 days
2. Test dataset was derived from the last 30 days until the end of the dataset

The model did forecasting based on last 30 days, and this experiment was repeated 10 times. This can be increased locally, and hyperparameter tuning is also recommended.

LSTM, accuracy 95.693%, time taken for 1 epoch 01:09

<img src="deep-learning-models-results/lstm.png" width="70%" align="">
