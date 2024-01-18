from lstm import LSTMForecast
from matplotlib import pyplot as plt
from optimization import optimize
import numpy as np
import pypfopt as ppf
import yfinance as yf
import pandas as pd
import torch
import random

# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)

tickers = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "ALL", "AMGN", "AMT", "AMZN",
    "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CHTR",
    "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DD",
    "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE", "GILD", "GM",
    "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI",
    "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL",
    "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SLB",
    "SO", "SPG", "T", "TGT", "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB",
    "V", "VZ", "WBA", "WFC", "WM", "WMT", "XOM"
]
tickers.sort()
forward = 1  # test result after 'forward' days
stock_data = yf.download(tickers, period="15y")
stock_data.dropna(how="all", inplace=True)
train_data = stock_data.iloc[:-forward]
test_prices = stock_data.iloc[-1]

lstm = LSTMForecast(tickers,
                    train_data,
                    lookback=10,
                    forward=forward,
                    n_nodes=5,
                    n_stack_layers=4,
                    learning_rate=0.0001,
                    n_epochs=200)
lstm.train()

mu_1 = pd.Series(lstm.predict_1step_ahead()[0] * 252 / forward, index=tickers)
mu_2 = ppf.expected_returns.capm_return(train_data["Close"],
                                        risk_free_rate=0.05)

# validate if ticker index is not match
for i in range(0, len(tickers)):
  if tickers[i] != mu_2.index[i]:
    raise ValueError("ticker index not match")

cov = ppf.risk_models.CovarianceShrinkage(train_data["Close"]).ledoit_wolf()
global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(cov)))
risks = np.arange(global_min_volatility + 0.01, 1, 0.01)
property_1_test = np.zeros(len(risks))
property_2_test = np.zeros(len(risks))
init_money = 1000

print("Optimizing money allocation...")
for i in range(0, len(risks)):
  weights_1 = pd.Series((optimize(mu_1.to_numpy(), cov.to_numpy(), risks[i])),
                        index=tickers)
  weights_2 = pd.Series((optimize(mu_2.to_numpy(), cov.to_numpy(), risks[i])),
                        index=tickers)

  property_1_before = 0
  property_2_before = 0
  last_price = train_data.iloc[-1]
  for ticker in tickers:
    property_1_before += last_price["Close"][ticker] * weights_1[ticker]
    property_2_before += last_price["Close"][ticker] * weights_2[ticker]

  # normalize
  for ticker in tickers:
    weights_1[ticker] *= init_money / property_1_before
    weights_2[ticker] *= init_money / property_2_before

  property_1_test[i] = 0
  property_2_test[i] = 0
  for ticker in tickers:
    property_1_test[i] += test_prices["Close"][ticker] * weights_1[ticker]
    property_2_test[i] += test_prices["Close"][ticker] * weights_2[ticker]

plt.plot(risks, property_1_test, label="LSTM optimization")
plt.plot(risks, property_2_test, label="Static optimization")
plt.legend()
plt.show()
