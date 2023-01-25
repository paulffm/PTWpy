# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from math import floor
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint
LeakyReLU = LeakyReLU(alpha=0.01)


'''lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=1e-2,
decay_steps=10000,
decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)'''

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

def neural_net(params, n_inputs, n_outputs):
    opt = Adam(lr=params['learning_rate'])
    nn = Sequential()
    nn.add(Dense(params['neurons'], input_shape=(n_inputs,), activation=params['activation']))
    if params['normalization'] > 0.5:
        nn.add(BatchNormalization())
    for i in range(params['layers1']):
        nn.add(Dense(params['neurons'], activation=params['activation']))
    if params['dropout'] > 0.5:
        nn.add(Dropout(params['dropout_rate'], seed=123))
    for i in range(params['layers2']):
        nn.add(Dense(params['neurons'], activation=params['activation']))
    nn.add(Dense(params['neurons'], activation=params['activation']))
    nn.add(Dense(n_outputs, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return nn

def scaling_method(method):
    if method == 'MinMax(0.1)':
        scaler_x_test = MinMaxScaler(feature_range=(0, 1))
        scaler_x_train = MinMaxScaler(feature_range=(0, 1))
        scaler_y_test = MinMaxScaler(feature_range=(0, 1))
        scaler_y_train = MinMaxScaler(feature_range=(0, 1))
    elif method == 'MinMax()':
        scaler_x_test = MinMaxScaler()
        scaler_x_train = MinMaxScaler()
        scaler_y_test = MinMaxScaler()
        scaler_y_train = MinMaxScaler()
    elif method == 'Standard':
        scaler_x_test = StandardScaler()
        scaler_x_train = StandardScaler()
        scaler_y_test = StandardScaler()
        scaler_y_train = StandardScaler()

    return scaler_x_train, scaler_x_test, scaler_y_train, scaler_y_test


def main():
    ### Parameter to choose:
    active_axis = 1

    # input: DES_POS, VEL_FFW, TORQUE_FFW: 0=from active axis, 1=from all axis
    input = 0

    # scaling: yes=1 no=0 -> yes, standard
    # Scaling method: 'MinMax(0.1)', 'MinMax()', 'Standard'
    scaling = 1
    method = 'Standard'

    # shifting: yes=1 no=0, window_size: 3 Benchmark
    # forw: if forward and backward shifting or only backwards: both=1 only backward=0
    shifting = 1
    step = 3
    forw = 1

    # which search method we use: 0=GridSearch, 1=RandomSearch, 2=BayesSearch
    # BayesSearch or RandomSearch prefered
    search_method = 2

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
                'forw': yes_no[forw],
                'search_method': search_list[search_method]}
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

    # filtering
    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]


    out_arr = [f'CURRENT|{active_axis}']
    X = data[input_list[input]]
    y = data[out_arr]

    # CV or split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # shifting forward and backward
    if shifting == 1:
        X, X_train = data_shift(X, X_train, step, forw)

    # Scaling:
    if scaling == 1:
        scaler_x, scaler_y, scaler_x_tr, scaler_y_tr = scaling_method(method)

        scaler_x = scaler_x.fit(X)
        scaler_y = scaler_y.fit(y)

        X = scaler_x.transform(X)
        y = scaler_y.transform(y)

        # scaling train
        scaler_x_tr = scaler_x_tr.fit(X_train)
        scaler_y_tr = scaler_y_tr.fit(y_train)

        X_train = scaler_x_tr.transform(X_train)
        y_train = scaler_y_tr.transform(y_train)


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
        'unit': randint(5, 100),
        'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
        'learning_rate': loguniform(0.001, 0.1),
        'layers1': [1, 2],
        'layers2': [1, 2],
        'dropout': uniform(0, 1),
        'dropout_rate': uniform(0, 0.3),
        'nb_epoch': randint(10, 100),
        'batch_size': randint(10, 100),
        'normalization': uniform(0, 1),
        'optimizerL': ['Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']}
        #'optimizerL': ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']}

    param_nn_bs = {
        'unit': Integer(5, 100),
        'activation': Categorical(['relu', 'sigmoid', 'tanh', 'elu', LeakyReLU]),
        'learning_rate': Real(0.001, 0.1),
        'layers1': Integer(1, 3),
        'layers2': Integer(1, 3),
        'dropout': Real(0, 1),
        'dropout_rate': Real(0, 0.3),
        'nb_epoch': Integer(10, 100),
        'batch_size': Integer(10, 100),
        'normalization': Real(0, 1),
        'optimizerL': ['Adam']}  #, 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta']}
        #'optimizerL': Categorical(['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])}
    


    print('build model')
    def nn_reg(learning_rate=0.01, unit=12, activation='relu', layers1=1, layers2=1, dropout=0.3, dropout_rate=0.23,
        normalization=0.3, nb_epoch=20, batch_size=20, optimizerL='Adam'):
        optimizerD = {'Adam': Adam(learning_rate=learning_rate), 'SGD': SGD(learning_rate=learning_rate),
                      'RMSprop': RMSprop(learning_rate=learning_rate), 'Adadelta': Adadelta(learning_rate=learning_rate),
                      'Adagrad': Adagrad(learning_rate=learning_rate), 'Adamax': Adamax(learning_rate=learning_rate),
                      'Nadam': Nadam(learning_rate=learning_rate), 'Ftrl': Ftrl(learning_rate=learning_rate)}

        opt = optimizerD[optimizerL]
        n_inputs = X_train.shape[1]
        nn = Sequential()
        nn.add(Dense(unit, input_shape=(n_inputs,), activation=activation))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(unit, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=123))
        for i in range(layers2):
            nn.add(Dense(unit, activation=activation))
        nn.add(Dense(1, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor="loss", patience=5, mode='auto', min_delta=0)
        history = nn.fit(X_train, y_train, batch_size=batch_size,
                         epochs=nb_epoch, callbacks=[early_stopping])
       
        return nn

    nn_reg_ = KerasRegressor(build_fn=nn_reg, verbose=0)

    # optimize:
    print('opt')
    n_iter_search = 15
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

    # inverse scaling
    if scaling == 1:

        '''X_train = scaler_x_tr.inverse_transform(X_train)
        X = scaler_x.inverse_transform(X)'''

        y = scaler_y.inverse_transform(y)
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred).reshape(-1, 1))


        # max error
        y_diff = np.sort(np.abs(np.asarray(y) - np.asarray(y_pred)), axis=0)
        y_max = y_diff[::-1]
        k = 4
        print('k max diff', y_max[:k])

    elif scaling == 0:

        # without scaling I can keep my index with this type of method to compute max diff
        y_diff_sort = (y - np.reshape(y_pred, (y_pred.shape[0], 1))).abs().sort_values(by=[f'CURRENT|{active_axis}'],
                                                                                       ascending=False)
        k = 4
        y_diff_k = y_diff_sort.iloc[:k]
        print('max diff', y_diff_k.iloc[0])
        print(f'{k} max diff: {y_diff_k}')

    # error values
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print('MSE:', mse)
    print('MAE:', mae)


if __name__ == '__main__':
        main()
