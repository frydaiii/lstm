import yfinance as yf
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from typing import List

tickers = ["SPY", "AAPL"]
stock_data = yf.download(" ".join(tickers), period="5y")
n_tickers = len(tickers)

device = 'mps:0' if torch.backends.mps.is_available(
) else 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeSeriesDataset(Dataset):

  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X[0])

  def __getitem__(self, i):
    return self.X[i], self.Y[i]


def prepare_data(df: pd.DataFrame, shift: int, scale: float):
  df.dropna(inplace=True)
  close_prices = stock_data["Close"].to_numpy()
  returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
  returns = np.append(np.zeros((1, returns.shape[1])), returns, axis=0)
  n_tickers = returns.shape[1]
  n_steps = returns.shape[0] - shift + 1
  # shifted_returns = np.zeros((n_tickers, n_steps, shift))
  # for i in range(0, n_tickers):
  #   for j in range(0, n_steps):
  #     shifted_returns[i][j] = returns.T[i][j:j + shift]

  shifted_returns = np.zeros((n_steps, shift, n_tickers))
  for i in range(0, n_steps):
    shifted_returns[i] = returns[i:i + shift, :]
  shifted_returns = np.float32(shifted_returns)

  X = shifted_returns[:, :-1, :]
  Y = shifted_returns[:, -1, :]
  split_index = int(n_steps * scale)
  X_train = X[:split_index]
  X_test = X[split_index:]

  Y_train = Y[:split_index]
  Y_test = Y[split_index:]

  # X_train = np.expand_dims(X_train, axis=3)
  # X_test = np.expand_dims(X_test, axis=3)

  # Y_train = np.expand_dims(Y_train, axis=2)
  # Y_test = np.expand_dims(Y_test, axis=2)

  print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
  return X_train, X_test, Y_train, Y_test


lookback = 14
split_scale = 0.95  # train/test scale
X_train, X_test, Y_train, Y_test = prepare_data(stock_data, lookback,
                                                split_scale)
train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

batch_size = 50

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LSTM(nn.Module):

  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(input_size,
                        hidden_size,
                        num_stacked_layers,
                        batch_first=True)

    self.fc = nn.Linear(hidden_size, n_tickers)

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size,
                     self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size,
                     self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out


model = LSTM(n_tickers, 5, 2)
model.to(device)
learning_rate = 0.0001
num_epochs = 1500
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch():
  model.train(True)
  # print(f'Epoch: {epoch + 1}')
  running_loss = 0.0

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99:  # print every 100 batches
      avg_loss_across_batches = running_loss / 100
      print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
                                              avg_loss_across_batches))
      running_loss = 0.0


def validate_one_epoch():
  model.train(False)
  running_loss = 0.0

  for batch_index, batch in enumerate(test_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
      output = model(x_batch)
      loss = loss_function(output, y_batch)
      running_loss += loss.item()

  avg_loss_across_batches = running_loss / len(test_loader)
  return avg_loss_across_batches

  # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
  # print('***************************************************')
  # print()


for epoch in range(num_epochs):
  train_one_epoch()
  avg_loss = validate_one_epoch()
  if epoch % 100 == 0:
    print('Val Loss of epoch {0}: {1:.3f}'.format(epoch, avg_loss))
    print('***************************************************')
    print()

with torch.no_grad():
  predicted = model(torch.from_numpy(X_train).to(device)).to('cpu').numpy()


def calculate_prices_from_predicted_returns(initial_prices, returns):
  stock_prices = [initial_prices[0]]

  for i in range(1, len(returns)):
    next_price = initial_prices[i - 1] * (1 + returns[i])
    stock_prices.append(next_price)

  return stock_prices


# plt.plot(Y_train.T[0], label=f'Actual Close {tickers[0]}')
# plt.plot(predicted.T[0], label=f'Predicted Close {tickers[0]}')
ticker_index = 1
S_0 = stock_data['Close'][tickers[ticker_index]].to_numpy()[-1]
S = stock_data['Close'][tickers[ticker_index]].to_numpy()[lookback - 1:]
plt.plot(S, label=f'Actual Close {tickers[0]}')
plt.plot(calculate_prices_from_predicted_returns(S, predicted.T[ticker_index]),
         label=f'Predicted Close {tickers[0]}')
plt.legend()
plt.show()



