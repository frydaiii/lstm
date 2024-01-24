from lstm import LSTMForecast
from matplotlib import pyplot as plt
from optimization import optimize
from utils import df2returns
from datetime import datetime
from utils import read_returns, normalize_return
import numpy as np
import pypfopt as ppf
import yfinance as yf
import pandas as pd
import torch
import random
import json

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

# LSTM returns
returns = {}
print(f"Forecasting returns...")
file = open('returns.txt', 'a')
start = datetime.strptime("18-01-2009", "%d-%m-%Y")
end = datetime.strptime("18-01-2024", "%d-%m-%Y")
for ticker in tickers:
  stock_data = yf.download(ticker, start=start, end=end)
  stock_data.dropna(how="all", inplace=True)
  train_data = stock_data.iloc[:-1]

  lstm = LSTMForecast(ticker,
                      train_data,
                      lookback=10,
                      batch_size=64,
                      n_nodes=10,
                      n_stack_layers=4,
                      learning_rate=1e-3,
                      n_epochs=2000)
  lstm.train()
  returns[ticker] = lstm.predict()
  print("{0}: {1:.4f}".format(ticker, returns[ticker]))
  file.write(ticker + ": " + str(returns[ticker]) + "\n")
print()
file.close()

returns = normalize_return(read_returns("returns_.txt"))
mu_1 = pd.Series(returns)


# CAPM returns
stock_data = yf.download(tickers, start=start, end=end)
train_data = stock_data.iloc[:-1]
mu_2 = ppf.expected_returns.capm_return(train_data["Close"],
                                        risk_free_rate=0.05)

# validate if ticker index is not match
for i in range(0, len(tickers)):
  if tickers[i] != mu_2.index[i]:
    raise ValueError("ticker index not match")

cov = ppf.risk_models.CovarianceShrinkage(train_data["Close"]).ledoit_wolf()
global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(cov)))
risks = np.arange(global_min_volatility + 0.01, 1, 0.01)
returns_portfolio_1 = np.zeros(len(risks))
returns_portfolio_2 = np.zeros(len(risks))

print("Optimizing money allocation...")
for i in range(0, len(risks)):
  weights_1 = pd.Series((optimize(mu_1.to_numpy(), cov.to_numpy(), risks[i])),
                        index=tickers)
  weights_2 = pd.Series((optimize(mu_2.to_numpy(), cov.to_numpy(), risks[i])),
                        index=tickers)

  for ticker in tickers:
    a = stock_data["Close"][ticker].to_numpy()[-1]
    b = stock_data["Close"][ticker].to_numpy()[-2]
    returns = a / b - 1
    returns_portfolio_1[i] += returns * weights_1[ticker]
    returns_portfolio_2[i] += returns * weights_2[ticker]

plt.plot(risks, returns_portfolio_1, label="LSTM optimization")
plt.plot(risks, returns_portfolio_2, label="Static optimization")
plt.legend()
plt.show()
