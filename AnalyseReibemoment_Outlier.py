import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from scipy.signal import butter, cheby1, filtfilt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def nn_reg(n_inputs, n_outputs):
    optimizer = Adam(learning_rate=0.001)
    nn = Sequential()
    nn.add(Dense(144, input_dim=n_inputs, activation='elu'))
    nn.add(Dense(144, activation='elu'))
    nn.add(Dense(144, activation='elu'))
    nn.add(Dense(144, activation='elu'))
    nn.add(Dense(144, activation='elu'))
    nn.add(Dense(1, activation='linear'))
    nn.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])

    return nn


def get_model(n_inputs, n_outputs):
    '''lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)'''
    lrelu = keras.layers.LeakyReLU(alpha=0.001)
    model = Sequential()
    model.add(Dense(100, activation=lrelu, input_dim=n_inputs))  # kernel_initializer='normal'
    model.add(Dense(50, activation=lrelu))
    model.add(Dense(50, activation=lrelu))
    model.add(Dense(30, activation=lrelu))
    model.add(Dense(n_outputs, activation='linear'))
    optimizer = optimizers.Adam(learning_rate=0.08)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    return model


def data_shift(X, window, forw, inp):
    # each look back/forward as another feature
    X_plc = X
    if forw == 1:
        for i in range(window):
            X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
            X_shift_fw = X.shift(periods=-(i + 1), fill_value=0)
            inp_bw = [x + f'_-{i + 1}' for x in inp]
            inp_fw = [x + f'_+{i + 1}' for x in inp]
            X_shift_bw.columns = inp_bw
            X_shift_fw.columns = inp_fw
            X_plc = pd.concat([X_plc, X_shift_bw, X_shift_fw], axis=1)
    else:
        for i in range(window):
            X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
            inp_bw = [x + f'_-{i + 1}' for x in inp]
            X_shift_bw.columns = inp_bw
            X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

    return X_plc


def scaling_method(method):
    if method == 'MinMax(0.1)':
        scaler_x_test = MinMaxScaler(feature_range=(0, 1))
        scaler_x_tr = MinMaxScaler(feature_range=(0, 1))
        scaler_y_test = MinMaxScaler(feature_range=(0, 1))
        scaler_y_tr = MinMaxScaler(feature_range=(0, 1))
    elif method == 'MinMax()':
        scaler_x_test = MinMaxScaler()
        scaler_x_tr = MinMaxScaler()
        scaler_y_test = MinMaxScaler()
        scaler_y_tr = MinMaxScaler()
    elif method == 'Standard':
        scaler_x_test = StandardScaler()
        scaler_x_tr = StandardScaler()
        scaler_y_test = StandardScaler()
        scaler_y_tr = StandardScaler()

    return scaler_x_tr, scaler_x_test, scaler_y_tr, scaler_y_test


def main():
    print('go')
    ### Parameter to choose:
    active_axis = 1

    # input: DES_POS, VEL_FFW, TORQUE_FFW: 0=from active axis, 1=from all axis -> 0 better
    input = 0

    # scaling: yes=1 no=0 -> for RF=0 for NN=1 and Standard
    # Scaling method: 'MinMax(0.1)', 'MinMax()', 'Standard'
    scaling = 0
    method = 'Standard'

    # shifting: yes=1 no=0, window_size: 3 Benchmark  ->nn=0 rf=1, step=3, forw=1
    # forw: if forward and backward shifting or only backwards: both=1 only backward=0
    shifting = 1
    step = 3
    forw = 1

    print('Specifications:')
    yes_no = ['no', 'yes']
    search_list = ['Grid', 'Random', 'Bayes']
    # cmd,cont and ctrl, cont schlechter als cmd, cont, ctrldiff2 und schlechter als cont
    inp = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}',
           f'CONT_DEV|{active_axis}', f'ENC1_POS|{active_axis}', f'ENC2_POS|{active_axis}']

    '''inp = [f'DES_POS|{active_axis}',f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}', 
           f'CONT_DEV|{active_axis}', f'CTRL_DIFF2|{active_axis}', f'CMD_SPEED|{active_axis}']'''
    # f'DES_POS|{active_axis}',
    # f'TORQUE|1']#, f'CTRL_DIFF2|{active_axis}']
    # f'CONT_DEV|{active_axis}']
    # f'CTRL_DIFF2|{active_axis}']
    # 'POWER|1']
    # f'ENC1_POS|{active_axis}', f'ENC2_POS|{active_axis}']
    # f'CONT_DEV|{active_axis}', 'CONT_DEV|2', 'CONT_DEV|3']
    # f'CTRL_DIFF|{active_axis}']
    # f'CONT_DEV|{active_axis}', f'CMD_SPEED|{active_axis}', f'CTRL_DIFF2|{active_axis}']
    # f'TORQUE|1']#, f'ENC1_POS|{active_axis}' - f'ENC2_POS|{active_axis}']
    # f'CONT_DEV|{active_axis}'], f'CMD_SPEED|{active_axis}', f'CTRL_DIFF2|{active_axis}']
    input_list = [inp]
    spec_dct = {'active axis': active_axis,
                'input': input_list[input],
                'scaling': yes_no[scaling],
                'scaling method': method,
                'shifting': yes_no[shifting],
                'step size': step,
                'forw': yes_no[forw]}
    print(spec_dct)

    ## data preparation
    feedVar = '/NC/_N_CH_GD9_ACX/SYG_S9|3'
    measurementVar = '/NC/_N_CH_GD9_ACX/SYG_S9|4'
    axisVar = '/NC/_N_CH_GD9_ACX/SYG_I9|2'
    loopVar1 = '/NC/_N_CH_GD9_ACX/SYG_I9|3'
    loopVar2 = '/NC/_N_CH_GD9_ACX/SYG_I9|4'
    loopVar3 = '/NC/_N_CH_GD9_ACX/SYG_I9|5'
    msgVar = '/Channel/ProgramInfo/msg|u1'
    blockVar = "/Channel/ProgramInfo/block|u1.2"

    # data: filtering better
    file = '2022-07-28_MRM_DMC850_20220509.MPF.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    # print('header', data.columns.values.tolist())
    data['DateTime'] = pd.to_datetime(data["_time"])
    data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(by="CYCLE")
    # print(data)

    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]

    # input, output
    out_arr = [f'CURRENT|{active_axis}']
    X = data[input_list[input]]
    y = data[out_arr]

    # kick enc1,2 and get only diff of them
    X['Enc_diff'] = X[f'ENC1_POS|{active_axis}'] - X[f'ENC2_POS|{active_axis}']
    X = X.drop([f'ENC1_POS|{active_axis}', f'ENC2_POS|{active_axis}'], axis=1)
    inp = X.columns.values.tolist()

    # filter signals
    order = 1
    b, a = butter(order, Wn=0.1, btype='lowpass')
    y_butter = filtfilt(b, a, y, axis=0)
    # X_butter = filtfilt(b, a, X, axis=0)
    print('vergleich', y, y_butter)

    # current over time plot to see how filter changes the current
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}'))  # , mode='markers', marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=data.index, y=y_butter.flatten(),
                             name=f'CURRENT|{active_axis} filtered'))  # , mode='markers',marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()

    # shifting data to include past values
    if shifting == 1:
        X = data_shift(X, step, forw, inp)
    print(X)

    # CV or split
    X_train, X_test, y_train, y_test = train_test_split(X, y_butter, test_size=0.4, random_state=1)

    # scaling the data
    if scaling == 1:
        scaler_x_tr, scaler_x, scaler_y_tr, scaler_y = scaling_method(method)

        X = scaler_x.fit_transform(X)
        y_butter = scaler_y.fit_transform(y_butter)

        # scaling train
        X_train = scaler_x_tr.fit_transform(X_train)
        y_train = scaler_y_tr.fit_transform(y_train)

    # to array for outlier detection
    '''X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X = np.asarray(X)

    # detect outliers 
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    #iso = IsolationForest()
    #yhat = iso.fit_predict(X)

    # select all rows that are not outliers and remove them
    mask = yhat != -1 
    X, y_butter = X[mask, :], y_butter[mask]

     # current over time plot to see how removed outliers change the current
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}'))  # , mode='markers', marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=data.index, y=y_butter.flatten(),
                             name=f'CURRENT|{active_axis} filtered'))  # , mode='markers',marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()

    #X = filtfilt(b, a, X, axis=0)
    #X_train = filtfilt(b, a, X_train, axis=0)'''

    # train and predict
    print('predict')
    # model = RandomForestRegressor(n_estimators=100, random_state=1, verbose=2, criterion="absolute_error")  # , criterion="absolute_error")
    model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=1000, verbosity=2)
    model.fit(np.asarray(X_train), y_train.flatten())  # , validation_split=0.2)
    y_pred = model.predict(np.asarray(X))

    # inverse transformation
    if scaling == 1:
        X_train = scaler_x_tr.inverse_transform(X_train)
        X = scaler_x.inverse_transform(X)
        y_butter = scaler_y.inverse_transform(y_butter)
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred).reshape(-1, 1))

    # max error
    y_diff = np.abs((y_butter) - (y_pred.reshape(-1, 1)))
    count = np.count_nonzero(y_diff > 0.1)
    print('Number of points with difference > 0.1:', count)

    idx = np.argsort(y_diff, axis=0)  # sorts along first axis (down)
    idx = idx[::-1]
    y_diff_sort = np.take_along_axis(y_diff, idx, axis=0)
    print('Max diff:', y_diff_sort[:1])
    idx = idx[:count].reshape(-1)

    # error metrics: MSE, MAE
    mse = mean_squared_error(y_butter, y_pred)
    mae = mean_absolute_error(y_butter, y_pred)
    print('MSE:', mse)
    print('MAE:', mae)

    # plots
    # y-y_pred plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_butter.flatten(), y=y_pred,
                             mode='markers', marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} Curve',
        xaxis_title=f'CURRENT|{active_axis}',
        yaxis_title=f'CURRENT|{active_axis} predicted',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()

    '''# pos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[f'ENC2_POS|{active_axis}'], y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}'))
    fig.add_trace(go.Scatter(x=data[f'ENC2_POS|{active_axis}'], y=y_pred,
                             name=f'CURRENT|{active_axis} pred'))

    fig.update_layout(
        title=f'CURRENT|{active_axis} over ENC2_POS|{active_axis}',
        xaxis_title=f'ENC2_POS|{active_axis}',
        yaxis_title=f'CURRENT|{active_axis}',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()'''

    '''# time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}')) #, mode='markers', marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=data.index, y=y_pred,
                             name=f'CURRENT|{active_axis} pred')) #, mode='markers',marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()'''

    # idx for max diff in plot
    y_diff_idx = (y.iloc[idx]).index
    y_mostd = y_butter[idx]

    # plot of 'normal' current, filtered current, predicted current and all points with diff > 0.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}'))
    fig.add_trace(go.Scatter(x=data.index, y=y_pred,
                             name=f'CURRENT|{active_axis} pred'))
    fig.add_trace(go.Scatter(x=data.index, y=y_butter.flatten(),
                             name=f'CURRENT|{active_axis} filtered'))
    fig.add_trace(go.Scatter(x=y_diff_idx, y=y_mostd.flatten(),
                             name=f'most difference', mode='markers',
                             marker=dict(size=10)))

    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()


if __name__ == '__main__':
    main()

