import numpy as np


def mse_loss(y: np.array, y_pred: np.array):
    return np.mean(np.square(y - y_pred))


def rmse_loss(y: np.array, y_pred: np.array):
    return np.sqrt(np.mean(np.square(y - y_pred)))
