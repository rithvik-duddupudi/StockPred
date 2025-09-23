<p align="center">
    <a href="#readme">
        <img alt="logo" width="50%" src="deep-learning-models-results/evolution-strategy.png">
    </a>
</p>
<p align="center">
  <a href="https://github.com/Twist3d-Fate/MLStockMarketPrediction/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/deeplearning-18--models-success.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/agent-24--models-success.svg"></a>
</p>

---

**ML-Stock-Market-Prediction**, a project that gathers machine learning and deep learning models for stock forecasting, including trading bots and simulations.

## Table of contents
  * [Models](#models)
  * [Agents](#agents)
  * [Simulations](#simulations)
  * [Frontend](#frontend)
  * [Results](#results)
    * [Agent Results](#agent-results)
    * [Model Results](#model-results)
    * [Simulation Results](#simulation-results)

## Contents

### Models

#### Deep-Learning Models
 1. LSTM
 2. LSTM Bidirectional
 3. LSTM 2-Path
 4. GRU
 5. GRU Bidirectional
 6. GRU 2-Path
 7. Vanilla
 8. Vanilla Bidirectional
 9. Vanilla 2-Path
 10. LSTM Seq2seq
 11. LSTM Bidirectional Seq2seq
 12. LSTM Seq2seq VAE
 13. GRU Seq2seq
 14. GRU Bidirectional Seq2seq
 15. GRU Seq2seq VAE
 16. Attention-is-all-you-Need
 17. CNN-Seq2seq
 18. Dilated-CNN-Seq2seq

#### Stacking Models
 1. Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor
 2. Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB

### Agents

1. Turtle-trading agent
2. Moving-average agent
3. Signal rolling agent
4. Policy-gradient agent
5. Q-learning agent
6. Evolution-strategy agent
7. Double Q-learning agent
8. Recurrent Q-learning agent
9. Double Recurrent Q-learning agent
10. Duel Q-learning agent
11. Double Duel Q-learning agent
12. Duel Recurrent Q-learning agent
13. Double Duel Recurrent Q-learning agent
14. Actor-critic agent
15. Actor-critic Duel agent
16. Actor-critic Recurrent agent
17. Actor-critic Duel Recurrent agent
18. Curiosity Q-learning agent
19. Recurrent Curiosity Q-learning agent
20. Duel Curiosity Q-learning agent
21. Neuro-evolution agent
22. Neuro-evolution with Novelty search agent
23. ABCD strategy agent
24. Evolution-strategy-bayesian agent 

### Simulations

1. Simple Monte Carlo, [monte-carlo-drift.ipynb](monte-carlo-simulation/monte-carlo-drift.ipynb)
2. Dynamic volatility Monte Carlo, [monte-carlo-dynamic-volatility.ipynb](monte-carlo-simulation/monte-carlo-dynamic-volatility.ipynb)
3. Drift Monte Carlo, [monte-carlo-drift.ipynb](monte-carlo-simulation/monte-carlo-drift.ipynb)
4. Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment, [multivariate-drift-monte-carlo.ipynb](monte-carlo-simulation/multivariate-drift-monte-carlo.ipynb)

### Frontend

I coded the [LSTM Recurrent Neural Network](deep-learning/1.lstm.ipynb) and [Simple Signal Rolling Agent](agent/simple-agent.ipynb) inside Tensorflow JS, and you can try it at [the StockPredict website](https://stockpredict999.netlify.app/). In the application, you can download any historical CSV and upload dynamically.

## Results

### Agent Results

**This agent only able to buy or sell 1 unit per transaction.**

1. Turtle-trading agent, [turtle-agent.ipynb](agents/turtle-agent.ipynb)

<img src="agents-results/turtle-agent.png" width="70%" align="">

2. Moving-average agent, [moving-average-agent.ipynb](agents/moving-average-agent.ipynb)

<img src="agents-results/moving-average-agent.png" width="70%" align="">

3. Signal rolling agent, [signal-rolling-agent.ipynb](agents/signal-rolling-agent.ipynb)

<img src="agents-results/signal-rolling-agent.png" width="70%" align="">

4. Policy-gradient agent, [policy-gradient-agent.ipynb](agents/policy-gradient-agent.ipynb)

<img src="agents-results/policy-gradient-agent.png" width="70%" align="">

5. Q-learning agent, [q-learning-agent.ipynb](agents/q-learning-agent.ipynb)

<img src="agents-results/q-learning-agent.png" width="70%" align="">

6. Evolution-strategy agent, [evolution-strategy-agent.ipynb](agents/evolution-strategy-agent.ipynb)

<img src="agents-results/evolution-strategy-agent.png" width="70%" align="">

7. Double Q-learning agent, [double-q-learning-agent.ipynb](agents/double-q-learning-agent.ipynb)

<img src="agents-results/double-q-learning.png" width="70%" align="">

8. Recurrent Q-learning agent, [recurrent-q-learning-agent.ipynb](agents/recurrent-q-learning-agent.ipynb)

<img src="agents-results/recurrent-q-learning.png" width="70%" align="">

9. Double Recurrent Q-learning agent, [double-recurrent-q-learning-agent.ipynb](agents/double-recurrent-q-learning-agent.ipynb)

<img src="agents-results/double-recurrent-q-learning.png" width="70%" align="">

10. Duel Q-learning agent, [duel-q-learning-agent.ipynb](agents/duel-q-learning-agent.ipynb)

<img src="agents-results/double-q-learning.png" width="70%" align="">

11. Double Duel Q-learning agent, [double-duel-q-learning-agent.ipynb](agents/double-duel-q-learning-agent.ipynb)

<img src="agents-results/double-duel-q-learning.png" width="70%" align="">

12. Duel Recurrent Q-learning agent, [duel-recurrent-q-learning-agent.ipynb](agents/duel-recurrent-q-learning-agent.ipynb)

<img src="agents-results/duel-recurrent-q-learning.png" width="70%" align="">

13. Double Duel Recurrent Q-learning agent, [double-duel-recurrent-q-learning-agent.ipynb](agents/double-duel-recurrent-q-learning-agent.ipynb)

<img src="agents-results/double-duel-recurrent-q-learning.png" width="70%" align="">

14. Actor-critic agent, [actor-critic-agent.ipynb](agents/actor-critic-agent.ipynb)

<img src="agents-results/actor-critic.png" width="70%" align="">

15. Actor-critic Duel agent, [actor-critic-duel-agent.ipynb](agents/actor-critic-duel-agent.ipynb)

<img src="agents-results/actor-critic-duel.png" width="70%" align="">

16. Actor-critic Recurrent agent, [actor-critic-recurrent-agent.ipynb](agents/actor-critic-recurrent-agent.ipynb)

<img src="agents-results/actor-critic-recurrent.png" width="70%" align="">

17. Actor-critic Duel Recurrent agent, [actor-critic-duel-recurrent-agent.ipynb](agents/actor-critic-duel-recurrent-agent.ipynb)

<img src="agents-results/actor-critic-duel-recurrent.png" width="70%" align="">

18. Curiosity Q-learning agent, [curiosity-q-learning-agent.ipynb](agents/curiosity-q-learning-agent.ipynb)

<img src="agents-results/curiosity-q-learning.png" width="70%" align="">

19. Recurrent Curiosity Q-learning agent, [recurrent-curiosity-q-learning.ipynb](agents/recurrent-curiosity-q-learning-agent.ipynb)

<img src="agents-results/recurrent-curiosity-q-learning.png" width="70%" align="">

20. Duel Curiosity Q-learning agent, [duel-curiosity-q-learning-agent.ipynb](agents/duel-curiosity-q-learning-agent.ipynb)

<img src="agents-results/duel-curiosity-q-learning.png" width="70%" align="">

21. Neuro-evolution agent, [neuro-evolution.ipynb](agents/neuro-evolution-agent.ipynb)

<img src="agents-results/neuro-evolution.png" width="70%" align="">

22. Neuro-evolution with Novelty search agent, [neuro-evolution-novelty-search.ipynb](agents/neuro-evolution-novelty-search-agent.ipynb)

<img src="agents-results/neuro-evolution-novelty-search.png" width="70%" align="">

23. ABCD strategy agent, [abcd-strategy.ipynb](agents/abcd-strategy-agent.ipynb)

<img src="agents-results/abcd-strategy.png" width="70%" align="">

24. Evolution-strategy bayesian agent, [evolution-strategy-bayesian-agent.ipynb](agents/evolution-strategy-bayesian-agent.ipynb)

<img src="agents-results/evolution-strategy-bayesian-agent.png" width="70%" align="">

### Model Results

For train-test split: 

1. Train dataset was derived from the starting timestamp until the last 30 days
2. Test dataset was derived from the last 30 days until the end of the dataset

The model did forecasting based on last 30 days, and this experiment was repeated 10 times. This can be increased locally, and hyperparameter tuning is also recommended.

1. LSTM, accuracy 95.693%, time taken for 1 epoch 01:09

<img src="deep-learning-models-results/lstm.png" width="70%" align="">

2. LSTM Bidirectional, accuracy 93.8%, time taken for 1 epoch 01:40

<img src="deep-learning-models-results/bidirectional-lstm.png" width="70%" align="">

3. LSTM 2-Path, accuracy 94.63%, time taken for 1 epoch 01:39

<img src="deep-learning-models-results/lstm-2path.png" width="70%" align="">

4. GRU, accuracy 94.63%, time taken for 1 epoch 02:10

<img src="deep-learning-models-results/gru.png" width="70%" align="">

5. GRU Bidirectional, accuracy 92.5673%, time taken for 1 epoch 01:40

<img src="deep-learning-models-results/bidirectional-gru.png" width="70%" align="">

6. GRU 2-Path, accuracy 93.2117%, time taken for 1 epoch 01:39

<img src="deep-learning-models-results/gru-2path.png" width="70%" align="">

7. Vanilla, accuracy 91.4686%, time taken for 1 epoch 00:52

<img src="deep-learning-models-results/vanilla.png" width="70%" align="">

8. Vanilla Bidirectional, accuracy 88.9927%, time taken for 1 epoch 01:06

<img src="deep-learning-models-results/bidirectional-vanilla.png" width="70%" align="">

9. Vanilla 2-Path, accuracy 91.5406%, time taken for 1 epoch 01:08

<img src="deep-learning-models-results/vanilla-2path.png" width="70%" align="">

10. LSTM Seq2seq, accuracy 94.9817%, time taken for 1 epoch 01:36

<img src="deep-learning-models-results/lstm-seq2seq.png" width="70%" align="">

11. LSTM Bidirectional Seq2seq, accuracy 94.517%, time taken for 1 epoch 02:30

<img src="deep-learning-models-results/bidirectional-lstm-seq2seq.png" width="70%" align="">

12. LSTM Seq2seq VAE, accuracy 95.4190%, time taken for 1 epoch 01:48

<img src="deep-learning-models-results/lstm-seq2seq-vae.png" width="70%" align="">

13. GRU Seq2seq, accuracy 90.8854%, time taken for 1 epoch 01:34

<img src="deep-learning-models-results/gru-seq2seq.png" width="70%" align="">

14. GRU Bidirectional Seq2seq, accuracy 67.9915%, time taken for 1 epoch 02:30

<img src="deep-learning-models-results/bidirectional-gru-seq2seq.png" width="70%" align="">

15. GRU Seq2seq VAE, accuracy 89.1321%, time taken for 1 epoch 01:48

<img src="deep-learning-models-results/gru-seq2seq-vae.png" width="70%" align="">

16. Attention Learning, accuracy 94.2482%, time taken for 1 epoch 01:41

<img src="deep-learning-models-results/attention-learning.png" width="70%" align="">

17. CNN-Seq2seq, accuracy 90.74%, time taken for 1 epoch 00:43

<img src="deep-learning-models-results/cnn-seq2seq.png" width="70%" align="">

18. Dilated-CNN-Seq2seq, accuracy 95.86%, time taken for 1 epoch 00:14

<img src="deep-learning-models-results/dilated-cnn-seq2seq.png" width="70%" align="">

19. Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + XGB

<img src="deep-learning-models-results/stack-rnn-arima-xgb.png" width="70%" align="">

20. Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB

<img src="deep-learning-models-results/stack-encoder-ensemble-xgb.png" width="70%" align="">

### Simulation Results

1. Simple Monte Carlo

<img src="monte-carlo-simulation/monte-carlo-simple.png" width="70%" align="">

2. Dynamic volatity Monte Carlo

<img src="monte-carlo-simulation/monte-carlo-dynamic-volatility.png" width="70%" align="">

3. Drift Monte Carlo

<img src="monte-carlo-simulation/monte-carlo-drift.png" width="70%" align="">

4. Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment

<img src="monte-carlo-simulation/multivariate-drift-monte-carlo.png" width="70%" align="">
