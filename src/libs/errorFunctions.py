import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

from libs.utils import *


def mse(trueVals, forecast):
    mse = mean_squared_error(trueVals, forecast)
    return mse


def rmse(trueVals, forecast):
    mse = mean_squared_error(trueVals, forecast)
    rmse = math.sqrt(mse)
    return rmse


def mae(trueVals, forecast):
    mae = mean_absolute_error(trueVals, forecast)
    return mae
