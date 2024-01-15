from scipy.optimize import minimize
import numpy as np


def mean_portfolio(weights: np.ndarray, *args):
  mu = args[0]
  return np.dot(mu.T, weights)


def portfolio_optimize_fun(weights: np.ndarray, *args):
  return -mean_portfolio(weights, *args)


def std_portfolio(weights: np.ndarray, cov: np.ndarray):
  return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))


def risk_constraint(weights: np.ndarray, risk: float, cov: np.ndarray):
  return risk - std_portfolio(weights, cov)


def optimize(mu: np.ndarray,
             cov: np.ndarray,
             risk: float,
             lower: int = 0,
             upper: int = 1):
  if mu.ndim != 1:
    raise ValueError("mean is not a 1-d array")
  if cov.ndim != 2:
    raise ValueError("cov is not a matrix")
  bounds = tuple((lower, upper) for _ in mu)
  init_w = np.ones(len(mu)) / len(mu)
  constraints = ({
      'type': 'ineq',
      'fun': risk_constraint,
      'args': (risk, cov)
  }, {
      'type': 'eq',
      'fun': lambda x: np.sum(x) - 1
  })

  result = minimize(portfolio_optimize_fun,
                    init_w,
                    args=(mu, cov),
                    bounds=bounds,
                    constraints=constraints)
  return result.x
