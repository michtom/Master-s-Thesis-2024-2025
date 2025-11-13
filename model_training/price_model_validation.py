from typing import Any

import pandas as pd
import numpy as np
from numpy import floating
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return np.mean(np.square(y_true - y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return np.mean(np.abs(y_true - y_pred))

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return (np.mean(calculate_single_return(y_true, y_pred)) / np.std(y_true, ddof=1))

def calculate_single_return(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    side = np.where(y_pred > 0, 1, -1)
    return side * y_true

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return r2_score(y_true, y_pred)

def calculate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    return pearsonr(y_true, y_pred).statistic


def _test():
    df = pd.read_csv(f'../data/market_data/btc_merged.csv')
    df['price_shift'] = df['price'].shift(-1)
    df['log_return'] = np.log(df['price_shift'] / df['price'])
    sample = df.dropna().head(30)['log_return'].to_numpy()
    dumb_buy = np.random.normal(0, 1, len(sample))
    print(sample)
    print(dumb_buy)
    print(f'mse: {np.sqrt(mean_squared_error(sample, dumb_buy))}')
    print(f'mae: {mean_absolute_error(sample, dumb_buy)}')
    print(f'dir_acc: {directional_accuracy(sample, dumb_buy)}')
    print(f'sharpe: {sharpe_ratio(sample, dumb_buy)}')
    print(f'r_squared: {calculate_r_squared(sample, dumb_buy)}')
    print(f'corr: {calculate_correlation(sample, dumb_buy)}')


if __name__ == '__main__':
    _test()