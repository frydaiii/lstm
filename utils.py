import pandas as pd
import numpy as np


def df2returns(df: pd.DataFrame):
  close_prices = df["Close"].to_numpy()
  returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
  returns = np.append(np.zeros((1, returns.shape[1])), returns, axis=0)
  return returns


def df2prices(df: pd.DataFrame):
  return df["Close"].to_numpy()


def read_returns(path: str):
  '''
  Read returns data from file.
  '''
  data_dict = {}

  with open(path, 'r') as file:
    for line in file:
      key, value = line.strip().split(': ')
      data_dict[key] = float(value)
  return data_dict


def normalize_return(returns: dict):
  '''
    If returns between +-5% then keep.
    Else if returns between +-10% then assign to +-5%.
    Else assign to 0.
    '''
  normalized_returns = {}

  for ticker, return_value in returns.items():
    if -0.05 <= return_value <= 0.05:
      normalized_returns[ticker] = return_value
    else:
      normalized_returns[ticker] = 0.0

  return normalized_returns
