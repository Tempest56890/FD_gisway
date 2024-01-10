import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape = (abs(y_predict - y_test) / y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    return mae, rmse, mape, r_2