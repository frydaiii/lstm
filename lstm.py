import yfinance as yf
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from typing import List
from sklearn.preprocessing import MinMaxScaler
from utils import df2returns


class TimeSeriesDataset(Dataset):

  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X[0])

  def __getitem__(self, i):
    return self.X[i], self.Y[i]


class LSTM(nn.Module):

  def __init__(self, input_size: int, hidden_size: int,
               num_stacked_layers: int, output_size: int, device: str):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(input_size,
                        hidden_size,
                        num_stacked_layers,
                        batch_first=True)

    self.fc = nn.Linear(hidden_size, output_size)
    self.device = device

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size,
                      self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size,
                      self.hidden_size).to(self.device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out
  # def forward(self, x):
  #   h0 = torch.zeros(self.num_stacked_layers, x.size(0),
  #                     self.hidden_size).requires_grad_()
  #   c0 = torch.zeros(self.num_stacked_layers, x.size(0),
  #                     self.hidden_size).requires_grad_()
  #   out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
  #   out = self.fc(out[:, -1, :])
  #   return out


class LSTMForecast(object):

  def __init__(self,
               tickers: List[str],
               stock_data: pd.DataFrame,
               lookback: int = 7,
               forward: int = 1,
               batch_size: int = 50,
               n_nodes: int = 5,
               n_stack_layers: int = 1,
               learning_rate: float = 0.0001,
               n_epochs: int = 1500):
    '''
    lookback: number of observations required to predict future values.
    forward: predict after "forward" steps.
    batch_size: size of each batch.
    n_nodes: number of nodes in hidden layer.
    n_stack_layers: number of stack layers.
    learning_rate: learning rate.
    n_epochs: number of epochs.
    '''
    # init data
    self.tickers = tickers
    self.stock_data = stock_data
    self.n_tickers = len(tickers)

    # init device
    self.device = 'mps:0' if torch.backends.mps.is_available(
    ) else 'cuda' if torch.cuda.is_available() else 'cpu'

    # init hyper parameters
    self.lookback = lookback
    self.forward = forward
    self.batch_size = batch_size
    self.n_epochs = n_epochs

    # init model
    self.model = LSTM(self.n_tickers, n_nodes, n_stack_layers, self.n_tickers, self.device)
    self.model.to(self.device)
    self.loss_function = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=learning_rate)

  def _prepare_data(self):
    prices = self.stock_data["Close"].to_numpy()
    n_tickers = prices.shape[1]
    n_interval = self.lookback + self.forward
    n_steps = prices.shape[0] - n_interval + 1

    X = np.zeros((n_steps, self.lookback, n_tickers))
    Y = np.zeros((n_steps, n_tickers))
    for i in range(0, n_steps):
      X[i] = prices[i:i + self.lookback, :]
      Y[i] = prices[i + n_interval - 1]
    X = np.float32(X)
    Y = np.float32(Y)

    # scale input
    scalers = {}
    for i in range(0, n_tickers):
      scalers[i] = MinMaxScaler(feature_range=(-1, 1))
      X[:, :, i] = scalers[i].fit_transform(X[:, :, i])
    # The target values are 2D arrays, which is easy to scale
    scalerY = MinMaxScaler(feature_range=(-1, 1))
    Y = scalerY.fit_transform(Y)

    # # flatten input
    # n_input = X.shape[1] * X.shape[2]
    # X = X.reshape((X.shape[0], n_input, 1))

    return X, Y

  def _train_one_epoch(self, loader):
    self.model.train(True)
    running_loss = 0.0

    for batch_index, batch in enumerate(loader):
      x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

      output = self.model(x_batch)
      loss = self.loss_function(output, y_batch)
      running_loss += loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if batch_index % 100 == 99:  # print every 100 batches
        avg_loss_across_batches = running_loss / 100
        print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
                                                avg_loss_across_batches))
        running_loss = 0.0

  def _validate_one_epoch(self, loader):
    self.model.train(False)
    running_loss = 0.0

    for _, batch in enumerate(loader):
      x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

      with torch.no_grad():
        output = self.model(x_batch)
        loss = self.loss_function(output, y_batch)
        running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(loader)
    return avg_loss_across_batches

  def train(self):
    X_train, Y_train = self._prepare_data()
    train_dataset = TimeSeriesDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True)

    for epoch in range(self.n_epochs):
      self._train_one_epoch(train_loader)
      avg_loss = self._validate_one_epoch(train_loader)
      if epoch % 100 == 0:
        print('Val Loss of epoch {0}: {1:.3f}'.format(epoch, avg_loss))
        print('***************************************************')
        print()

  def predict_1step_ahead(self):
    # _todo rescale from (-1,1)
    prices = self.stock_data["Close"].to_numpy()
    X = np.zeros((1, self.lookback, self.n_tickers))
    X[0] = prices[-self.lookback:, :]
    X = np.float32(X)
    with torch.no_grad():
      predicted = self.model(torch.from_numpy(X).to(
          self.device)).to('cpu').numpy()

    last_prices = X[-1]
    return predicted / last_prices - 1

  def _pred_returns_to_pred_prices(self, initial_prices, returns):
    stock_prices = [initial_prices[0]]

    for i in range(1, len(returns)):
      next_price = initial_prices[i - 1] * (1 + returns[i])
      stock_prices.append(next_price)

    return stock_prices

  def plot_train_result(self, ticker_index=0):
    '''
    plot actual prices and predicted prices of a ticker in ticker list.
    '''
    ticker = self.tickers[ticker_index]
    X, Y = self._prepare_data()
    with torch.no_grad():
      predicted = self.model(torch.from_numpy(X).to(
          self.device)).to('cpu').numpy()
    plt.plot(Y[:, ticker_index], label=f'Actual Returns {ticker}')
    plt.plot(predicted[:, ticker_index], label=f'Predicted Returns {ticker}')
    plt.legend()
    plt.show()
