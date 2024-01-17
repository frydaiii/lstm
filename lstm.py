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
from copy import deepcopy as dc


class TimeSeriesDataset(Dataset):

  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.Y[i]


class LSTM(nn.Module):

  def __init__(self, hidden_size: int,
               num_stacked_layers: int, device: str):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(1,
                        hidden_size,
                        num_stacked_layers,
                        batch_first=True)

    self.fc = nn.Linear(hidden_size, 1)
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
               ticker: str,
               stock_data: pd.DataFrame,
               lookback: int = 7,
               batch_size: int = 16,
               n_nodes: int = 5,
               n_stack_layers: int = 1,
               learning_rate: float = 0.001,
               n_epochs: int = 20):
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
    self.ticker = ticker
    self.stock_data = stock_data

    # init device
    self.device = 'mps:0' if torch.backends.mps.is_available(
    ) else 'cuda' if torch.cuda.is_available() else 'cpu'

    # init hyper parameters
    self.lookback = lookback
    self.batch_size = batch_size
    self.n_epochs = n_epochs

    # init model
    self.model = LSTM(n_nodes, n_stack_layers, self.device)
    self.model.to(self.device)
    self.loss_function = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=learning_rate)

  def _prepare_data(self):
    df = pd.DataFrame()
    df["Close"] = self.stock_data["Close"]

    for i in range(1, self.lookback + 1):
      df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)

    df_np = df.to_numpy()
    self.scaler = MinMaxScaler(feature_range=(-1, 1))
    df_np = self.scaler.fit_transform(df_np)
    X = df_np[:, :-1]
    Y = df_np[:, -1]
    X = dc(np.flip(X, axis=1))
    X = X.reshape(-1, self.lookback, 1)
    Y = Y.reshape(-1, 1)
    return torch.tensor(X).float(), torch.tensor(Y).float()

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
      if epoch % 10 == 0:
        print('Val Loss of epoch {0}: {1:.3f}'.format(epoch, avg_loss))
        print('***************************************************')
        print()

  def scale_normal(self, data: np.ndarray):
    '''
    Rescale data to normal from (-1,1).
    '''
    origin = data.flatten()

    dummies = np.zeros((len(data), self.lookback + 1))
    dummies[:, 0] = origin
    dummies = self.scaler.inverse_transform(dummies)

    origin = dc(dummies[:, 0])
    return origin

  def predict(self):
    '''
    Predict 1-step ahead.
    '''
    prices = self.stock_data["Close"].to_numpy()
    X = np.zeros((1, self.lookback, self.n_tickers))
    X[0] = prices[-self.lookback:, :]
    X = np.float32(X)
    with torch.no_grad():
      predicted = self.model(torch.from_numpy(X).to(
          self.device)).to('cpu').numpy()

    last_prices = X[-1]
    return predicted / last_prices - 1

  def plot_train_result(self):
    '''
    Plot actual prices and predicted prices of a ticker in ticker list.
    '''
    X, Y = self._prepare_data()
    with torch.no_grad():
      predicted = self.model(X.to(self.device)).to('cpu').numpy()
    plt.plot(self.scale_normal(Y), label=f'Actual Returns')
    plt.plot(self.scale_normal(predicted), label=f'Predicted Returns')
    plt.legend()
    plt.show()
