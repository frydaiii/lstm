from lstm import LSTMForecast

tickers = ["SPY", "AAPL"]
lstm = LSTMForecast(tickers)
lstm.train()
lstm.plot_train_result()