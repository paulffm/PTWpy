# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
import plotly.graph_objects as go
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adagrad, Adamax, Nadam, Ftrl
# from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import LeakyReLU
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint
from scipy.signal import butter, cheby1, filtfilt
from keras import initializers

LeakyReLU = LeakyReLU(alpha=0.01)


def scaling_method(method):
    if method == 'MinMax(-1.1)':
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

    elif method == 'MinMax()':
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

    elif method == 'Standard':
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

    return scaler_x, scaler_y


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


def main():
    print('Go')
    '''if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")'''
    ### Parameter to choose:
    active_axis = 1

    ### Parameter to choose:
    active_axis = 1

    # input: DES_POS, VEL_FFW, TORQUE_FFW: 0=from active axis, 1=from all axis -> 0 better
    input = 0

    # scaling: yes=1 no=0 -> for RF=0 for NN=1 and Standard
    # Scaling method: 'MinMax(-1.1)', 'MinMax()', 'Standard'
    scaling = 1
    method = 'Standard'

    # which search method we use: 0=GridSearch, 1=RandomSearch, 2=BayesSearch
    # BayesSearch or RandomSearch prefered
    search_method = 1

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

    input_list = [inp]
    spec_dct = {'active axis': active_axis,
                'input': input_list[input],
                'scaling': yes_no[scaling],
                'scaling method': method,
                'shifting': yes_no[shifting],
                'step size': step,
                'forw': yes_no[forw],
                'search_method': search_list[search_method]
                }

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

    # scaling
    if scaling == 1:
        scaler_x_r = RobustScaler()
        scaler_y_r = RobustScaler()
        X = scaler_x_r.fit_transform(X)
        y_butter = scaler_y_r.fit_transform(y_butter)

        scaler_x_m, scaler_y_m = scaling_method(method)
        X = scaler_x_m.fit_transform(X)
        y_butter = scaler_y_m.fit_transform(y_butter)

    # CV or split
    X_train, X_test, y_train, y_test = train_test_split(X, y_butter, test_size=0.4, random_state=1)

    # params nn
    param_nn_grid = {
        'unit': [7, 10, 12, 17, 20, 25],
        'activation': ['relu', 'tanh', 'selu', 'elu'],
        'learning_rate': [0.001, 0.005, 0.01, 0.1],
        'batch_size': [10, 15, 20, 30, 40],
        'epochs': [10, 15, 20, 30, 50],
        'layers1': [1, 2],
        'layers2': [1, 2],
        'dropout': [0, 0.1, 0.3, 0.5, 0.7],
        'dropout_rate': [0, 0.1, 0.3]}

    param_nn_rand = {
        'unit1': randint(50, 250),
        'unit2': randint(50, 250),
        'unit3': randint(50, 2500),
        'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
        'learning_rate': loguniform(0.001, 0.1),
        'layers1': [1, 2],
        'layers2': [0, 2],
        'nb_epoch': randint(10, 100),
        'batch_size': randint(10, 100),
        'kernel_initializer': ['he_uniform', 'glorot_uniform']
        # 'normalization': uniform(0, 1),
        # 'optimizerL': ['Adam', 'SGD', 'Adagrad', 'Adamax', 'Adadelta', 'Nadam', 'Ftrl']}}
        # 'optimizerL': ['RMSprop', 'Nadam', 'Ftrl']}
    }

    param_nn_bs = {
        'unit1': Integer(50, 250),
        'unit2': Integer(50, 250),
        'unit3': Integer(50, 2500),
        'activation': Categorical(['relu', 'sigmoid', 'tanh', 'elu', LeakyReLU]),
        'learning_rate': Real(0.001, 0.1),
        'layers1': Integer(1, 2),
        'layers2': Integer(0, 2),
        'nb_epoch': Integer(80, 150),
        'batch_size': Integer(80, 150),
        # 'normalization': Real(0, 1),
        # 'optimizerL': ['Adam', 'SGD', 'Adagrad', 'Adamax', 'Adadelta', 'Nadam', 'Ftrl']}  #, 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']}
        # 'optimizerL': Categorical(['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])}
    }
    print('build model')

    def nn_reg(learning_rate=0.01, unit1=12, unit2=12, unit3=12, activation='relu', layers1=1, layers2=1,
               normalization=0.3, nb_epoch=20, batch_size=20, optimizerL='Adam', kernel_initializer='he_uniform'):
        optimizerD = {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate),
                      'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate),
                      'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}

        opt = optimizerD[optimizerL]
        n_inputs = X_train.shape[1]
        nn = Sequential()
        nn.add(Dense(unit1, input_shape=(n_inputs,), activation=activation, kernel_initializer=kernel_initializer,
                     bias_initializer=initializers.Constant(0.01)))
        # if normalization > 0.5:
        #   nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(unit2, activation=activation))
            nn.add(Dropout(0.2, seed=123))
        for i in range(layers2):
            nn.add(Dense(unit3, activation=activation))
        nn.add(Dense(1, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor="loss", patience=8, mode='auto', min_delta=0)
        history = nn.fit(X_train, y_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping], validation_split=0.15)

        return nn

    nn_reg_ = KerasRegressor(build_fn=nn_reg, verbose=0)

    # optimize:
    print('opt')
    n_iter_search = 40
    if search_method == 0:
        search_reg = GridSearchCV(nn_reg, param_nn_grid, cv=5, scoring='accuracy')
    elif search_method == 1:
        search_reg = RandomizedSearchCV(nn_reg_, param_nn_rand, cv=5, scoring='max_error', n_iter=n_iter_search)
    elif search_method == 2:
        search_reg = BayesSearchCV(nn_reg_, param_nn_bs, n_iter=n_iter_search, scoring='max_error', cv=5)

    search_reg.fit(X_train, np.ravel(y_train))
    print(search_reg.best_params_)
    print(f'best score: {search_reg.best_score_}')
    # test prediction
    y_pred = search_reg.predict(X)

    # inverse transformation
    if scaling == 1:
        y_butter = scaler_y_m.inverse_transform(y_butter)
        y_butter = scaler_y_r.inverse_transform(y_butter)

        y_pred = scaler_y_m.inverse_transform(np.asarray(y_pred).reshape(-1, 1))
        y_pred = scaler_y_r.inverse_transform(y_pred)

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
    fig.add_trace(go.Scatter(x=y_butter.flatten(), y=y_pred.flatten(),
                             mode='markers', marker=dict(size=3)))
    fig.update_layout(
        title=f'CURRENT|{active_axis} Curve',
        xaxis_title=f'CURRENT|{active_axis}',
        yaxis_title=f'CURRENT|{active_axis} predicted',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()

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