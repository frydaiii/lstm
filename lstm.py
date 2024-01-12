import yfinance as yf
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from typing import List


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


class LSTMForecast(object):

  def __init__(self,
               tickers: List[str],
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
    self.stock_data = yf.download(" ".join(tickers), period="5y")
    self.n_tickers = len(tickers)

    # init device
    self.device = 'mps:0' if torch.backends.mps.is_available(
    ) else 'cuda' if torch.cuda.is_available() else 'cpu'

    # init hyper parameters
    self.lookback = lookback
    self.forward = forward
    self.batch_size = batch_size
    self.n_nodes = n_nodes
    self.n_stack_layers = n_stack_layers
    self.n_epochs = n_epochs

    # init model
    self.model = LSTM(self.n_tickers, 5, 2, self.n_tickers, self.device)
    self.model.to(self.device)
    self.loss_function = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

  def _prepare_data(self):
    '''
    convert data frame to returns matrix with shape (n_steps x lookback x n_tickers)
    '''
    self.stock_data.dropna(inplace=True)
    close_prices = self.stock_data["Close"].to_numpy()
    returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
    returns = np.append(np.zeros((1, returns.shape[1])), returns, axis=0)
    n_tickers = returns.shape[1]
    n_steps = returns.shape[0] - self.lookback + 1

    shifted_returns = np.zeros((n_steps, self.lookback, n_tickers))
    for i in range(0, n_steps):
      shifted_returns[i] = returns[i:i + self.lookback, :]
    shifted_returns = np.float32(shifted_returns)

    X = shifted_returns[:, :-1, :]
    Y = shifted_returns[:, -1, :]
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

  def predict_1step_ahead(self, data: pd.DataFrame):
    pass
    # with torch.no_grad():
    #   predicted = self.model(torch.from_numpy(X_train).to(self.device)).to('cpu').numpy()

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
    S = self.stock_data['Close'][ticker].to_numpy()[self.lookback - 1:]
    X_train, _ = self._prepare_data()
    with torch.no_grad():
      predicted = self.model(torch.from_numpy(X_train).to(
          self.device)).to('cpu').numpy()
    plt.plot(S, label=f'Actual Close {ticker}')
    plt.plot(self._pred_returns_to_pred_prices(S, predicted.T[ticker_index]),
             label=f'Predicted Close {ticker}')
    plt.legend()
    plt.show()
