from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from keras.layers import Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import itertools
import random
from skopt.space import Real, Categorical, Integer

def shift4(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def splitDict(d, n):
    i = iter(d.items())  # alternatively, i = d.iteritems() works in Python 2
    d1 = dict(itertools.islice(i, n))  # grab first n items
    d2 = dict(i)  # grab the rest
    return d1, d2

def main():

    '''feedVar = '/NC/_N_CH_GD9_ACX/SYG_S9|3'
    measurementVar = '/NC/_N_CH_GD9_ACX/SYG_S9|4'
    axisVar = '/NC/_N_CH_GD9_ACX/SYG_I9|2'
    loopVar1 = '/NC/_N_CH_GD9_ACX/SYG_I9|3'
    loopVar2 = '/NC/_N_CH_GD9_ACX/SYG_I9|4'
    loopVar3 = '/NC/_N_CH_GD9_ACX/SYG_I9|5'
    msgVar = '/Channel/ProgramInfo/msg|u1'
    blockVar = "/Channel/ProgramInfo/block|u1.2"

    # data
    file = r'/Users/paulheller/PycharmProjects/PTWKohn/Data/2022-07-28_MRM_DMC850_20220509.MPF.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    #print('header', dataFilter.columns.values.tolist())
    data['DateTime'] = pd.to_datetime(data["_time"])
    data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(by="CYCLE")
    print(data['DateTime'])
    print('header name:', data.columns.values.tolist())

    active_axis = 1
    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"

    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]

    inp = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}']
    out_arr = [f'CURRENT|{active_axis}']

    X = data[inp]
    print('X_1', X)
    y = data[out_arr]

    n = X.shape[0]
    t_pd = np.linspace(0, n / 3600, n)[:, np.newaxis]
    t_pd = pd.DataFrame(t_pd, columns=['t'])
    print('t_pd', t_pd)
    X = pd.concat([X.reset_index(drop=True), t_pd.reset_index(drop=True)], axis=1)
    print('X', X)'''





    '''a = np.ones((5, 1)) + 3
    b = np.zeros((5, 1)) + 1
    a[0] = 25
    a[1] = 70
    a[4] = 100
    a = a.reshape(-1, 1)
    print(a)
    c = shift5(a, 2, fill_value=0)
    d = shift5(a, -2, fill_value=0)
    print(c, d)
    a = np.concatenate((a, c, d), axis=1)
    print(a)'''

    '''d = {"bootstrap": Categorical([True, False]),
                    "max_depth": Integer(4, 14),
                    "max_features": Categorical(['auto', 'sqrt', 'log2']),
                    "min_samples_leaf": Integer(2, 6),
                    "min_samples_split": Integer(2, 6),
                    "n_estimators": Integer(100, 300),
                    "criterion": Categorical(['squared_error', 'absolute_error'])}
    d1, d2 = splitDict(d, 6)
    print(d1)
    print('d2', d2)
    print(d1['max_depth'])
    print(d1['max_depth'])
    print(d1[0])'''
    '''a = [1, 2, 3, 4, 10]
    b = a[:-2]
    c = a[-2:]
    print(b)
    print(c)'''
    x = np.array([1, 5, 0, 4, 2, 8])
    ind = np.argsort(x, axis=0)  # sorts along first axis (down)
    print(ind)
    print('ind', x[ind])
    ind = ind[::-1]
    print(ind)
    y = np.take_along_axis(x, ind, axis=0)
    print(y)
if __name__ == '__main__':
    main()