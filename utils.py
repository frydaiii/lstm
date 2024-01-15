import pandas as pd
import numpy as np


def df2returns(df: pd.DataFrame):
  close_prices = df["Close"].to_numpy()
  returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
  returns = np.append(np.zeros((1, returns.shape[1])), returns, axis=0)
  return returns

def df2prices(df: pd.DataFrame):
  return df["Close"].to_numpy()