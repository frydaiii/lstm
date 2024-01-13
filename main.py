from lstm import LSTMForecast

tickers = ["SPY", "AAPL"]
lstm = LSTMForecast(tickers, lookback=10, forward=6)
lstm.train()
# lstm.plot_train_result()
print(lstm.predict_1step_ahead())