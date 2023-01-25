
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn import svm

def scaling(method):
    if method == 'MinMax(0,1)':
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_x_tr = MinMaxScaler(feature_range=(0, 1))
        scaler_y_tr = MinMaxScaler(feature_range=(0, 1))
    elif method == 'MinMax()':
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x_tr = MinMaxScaler()
        scaler_y_tr = MinMaxScaler()
    elif method == 'Standard':
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x_tr = StandardScaler()
        scaler_y_tr = StandardScaler()

    return scaler_x, scaler_y, scaler_x_tr, scaler_y_tr

def build_model(n_inputs, n_features):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def data_shift(X, X_train, window, forw):
    X_plc = X
    idx_train = np.asarray(X_train.index)
    # each look back/forward as another feature
    if forw == 1:
        for i in range(window):
            X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
            X_shift_fw = X.shift(periods=-(i + 1), fill_value=0)
            X_train_shift_bw = X_shift_bw.loc[idx_train]
            X_train_shift_fw = X_shift_fw.loc[idx_train]
            X_train = pd.concat([X_train, X_train_shift_bw, X_train_shift_fw], axis=1)
            X_plc = pd.concat([X_plc, X_shift_bw, X_shift_fw], axis=1)
    else:
        for i in range(window):
            X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
            X_train_shift_bw = X_shift_bw.loc[idx_train]
            X_train = pd.concat([X_train, X_train_shift_bw], axis=1)
            X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

    return X_plc, X_train


def preprocess_multistep_lstm(sequence, n_steps_in, n_steps_out, n_features=1):
    # predicts n_steps_out in future
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    return X, y

def create_dataset_regr(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

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
    ### Parameter to choose:
    active_axis = 1

    # input: DES_POS, VEL_FFW, TORQUE_FFW: 0=from active axis, 1=from all axis
    input = 0

    # scaling: yes=1 no=0 ->
    # Scaling method: 'MinMax(0.1)', 'MinMax()', 'Standard'
    scaling = 1
    method = 'Standard'

    # shifting: yes=1 no=0, window_size: 3 Benchmark  ->1 makes max difference better
    # forw: if forward and backward shifting or only backwards: both=1 only backward=0
    shifting = 1
    step = 3
    forw = 1

    print('Specifications:')
    yes_no = ['no', 'yes']
    search_list = ['Grid', 'Random', 'Bayes']
    inp = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}']
    inp_all_axis = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}',
                    f'DES_POS|2', f'VEL_FFW|2', f'TORQUE_FFW|2',
                    f'DES_POS|3', f'VEL_FFW|3', f'TORQUE_FFW|3']
    input_list = [inp, inp_all_axis]
    spec_dct = {'active axis': active_axis,
                'input': input_list[input],
                'scaling': yes_no[scaling],
                'scaling method': scaling_method,
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

    # data
    file = r'/Users/paulheller/PycharmProjects/PTWPy/PTWKohn/Data/2022-07-28_MRM_DMC850_20220509.MPF.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    print('header', data.columns.values.tolist())
    data['DateTime'] = pd.to_datetime(data["_time"])
    data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(by="CYCLE")

    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]

    out_arr = [f'CURRENT|{active_axis}']
    X = data[input_list[input]]
    y = data[out_arr]

    time_steps = 60
    # shape: Xp: lenX, time_steps, 1=output yp: lenY, 1
    Xp, yp = create_dataset_regr(X, y, time_steps=time_steps)

    # split for fitting and testing
    test_days = round(0.4 * X.shape[0])
    Xp_train, yp_train = Xp[:-test_days], yp[:-test_days]
    Xp_test, yp_test = Xp[-test_days:], yp[-test_days:]

    # shifting forward and backward: dont make sense?
    '''if shifting == 1:
        X, X_train = data_shift(X, X_train, step, forw)'''

    # Scaling:
    if scaling == 1:
        scaler_xtr, scaler_x, scaler_ytr, scaler_y = scaling_method(method)

        scaler_x = scaler_x.fit(Xp_test)
        scaler_y = scaler_y.fit(yp_test)

        Xp_test = scaler_x.transform(Xp_test)
        yp_test = scaler_y.transform(yp_test)


        # scaling train
        scaler_xtr = scaler_xtr.fit(Xp_train)
        scaler_ytr = scaler_ytr.fit(yp_train)

        Xp_train = scaler_xtr.transform(Xp_train)
        yp_train = scaler_ytr.transform(yp_train)

    # build LSTM
    n_inputs = Xp_train.shape[1]
    n_features = Xp_train.shape[2]
    nb_epoch = 10
    batch_size = 128 #32
    model = build_model(n_inputs, n_features)
    early_stopping = EarlyStopping(monitor="loss", patience=6, mode='auto', min_delta=0)
    history = model.fit(Xp_train, yp_train, batch_size=batch_size,
                        epochs=nb_epoch, callbacks=[early_stopping])


    ytr_pred = model.predict(Xp_train)
    y_pred = model.predict(Xp_test)

    if scaling == 1:
        yp_train = scaler_ytr.inverse_transform(yp_train)
        ytr_pred = scaler_ytr.inverse_transform(ytr_pred)

        yp_test = scaler_y.inverse_transform(yp_test)
        y_pred = scaler_y.inverse_transform(y_pred)

        y_pred = y_pred.reshape(-1, 1)
        ytr_pred = ytr_pred.reshape(-1, 1)


    # error
    print('MSE train:', mean_squared_error(yp_train, ytr_pred))
    print('MAE train:', mean_absolute_error(yp_train, ytr_pred))
    print('Max error train', max_error(yp_train, ytr_pred))

    print('MSE test:', mean_squared_error(yp_test, y_pred))
    print('MAE test:', mean_absolute_error(yp_test, y_pred))
    print('Max error test', max_error(yp_test, y_pred))


    y_diff_test = np.sort(np.abs(np.asarray(yp_test) - np.asarray(y_pred)), axis=0)
    y_max_test = y_diff_test[::-1]
    print('sorted test', y_max_test)
    print('max diff test', y_max_test[0])

    y_diff_train = np.sort(np.abs(np.asarray(yp_train) - np.asarray(ytr_pred)), axis=0)
    y_max_train = y_diff_train[::-1]
    print('sorted train', y_max_train)
    print('max diff train', y_max_train[0])

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()



    '''# y-y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y[f'CURRENT|{active_axis}'], y=y_pred,
                             mode='markers', marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} Curve',
        xaxis_title=f'CURRENT|{active_axis}',
        yaxis_title=f'CURRENT|{active_axis} predicted',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()


    # pos
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
    fig.show()


    # time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}', mode='markers',
                             marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=data.index, y=y_pred,
                             name=f'CURRENT|{active_axis} pred', mode='markers',
                             marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()'''


    '''fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=y[f'CURRENT|{active_axis}'],
                             name=f'CURRENT|{active_axis}'))
    fig.add_trace(go.Scatter(x=data.index, y=y_pred,
                             name=f'CURRENT|{active_axis} pred'))
    fig.add_trace(go.Scatter(x=y_diff_k.index, y=y_mostd[f'CURRENT|{active_axis}'],
                             name=f'most difference', mode='markers',
                             marker=dict(size=10)))

    fig.update_layout(
        title=f'CURRENT|{active_axis} over time',
        yaxis_title=f'CURRENT|{active_axis}',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()'''


if __name__ == '__main__':
    main()

    # transformation
    ''' X = np.sqrt(np.log(X))
    X = X - X.shift()
    X.dropna(inplace=True)


    y_or = y
    y_tf = np.sqrt(np.log(y))
    print(y_tf)
    y_tf_diff = y_tf - y_tf.shift()
    y = y_tf_diff
    y.dropna(inplace=True)
    print(y)
    #y = y.iloc[1:]

    X_train_tf = np.sqrt(np.log(X_train))
    print('x_tf', X_train)
    X_train_diff = X_train_tf - X_train_tf.shift()
    X_train = X_train_diff
    X_train.dropna(inplace=True)
    print('x', X_train)

    y_train_tf = np.sqrt(np.log(y_train))
    y_train_diff = y_train_tf - y_train_tf.shift()
    print('ytf', y_train_tf)
    y_train = y_train_diff
    y_train.dropna(inplace=True)
    #y_train.iloc[1:]
    print('ytf', y_train)'''

    # inverse scaling
    '''X_train = scaler_x_tr.inverse_transform(X_train)
    X = scaler_x.inverse_transform(X)
    #y = scaler_y.inverse_transform(y)
    y_pred = y_pred.reshape(-1, 1)
    #y_pred = scaler_y.inverse_transform(y_pred)'''

    # inverse transformation
    '''y_pred = pd.DataFrame(y_pred[:, 0], y_or.index, columns=['target'])
    y_pred['target'] = y_pred['target'] + y_tf.shift()
    y_pred = (y_pred ** 2)
    y_pred = np.exp(y_pred)
    y = y_or'''