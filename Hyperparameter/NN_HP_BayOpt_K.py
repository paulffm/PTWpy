# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, KFold
from keras.layers import LeakyReLU

LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)


'''def get_model(n_inputs, n_outputs):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    lrelu = keras.layers.LeakyReLU(alpha=0.001)
    model = Sequential()
    model.add(Dense(100, activation=lrelu, input_dim=n_inputs)) # kernel_initializer='normal'
    model.add(Dense(50, activation=lrelu))
    model.add(Dense(50, activation=lrelu))
    model.add(Dense(30, activation=lrelu))
    model.add(Dense(n_outputs, activation='linear'))
    optimizer = optimizers.Adam(learning_rate=0.08)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    return model'''


'''def nn_cv(neurons, activation, optimizer, learning_rate, batch_size, epochs,
             layers1, layers2, normalization, dropout, dropout_rate, X, y):
    n_inputs = X.shape[1]
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
    optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                  'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                  'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                  'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']
    neurons = round(neurons)
    activation = activationL[round(activation)]
    batch_size = round(batch_size)
    epochs = round(epochs)

    def nn():
        opt = Adam(lr=learning_rate)
        nn = Sequential()
        nn.add(Dense(neurons, input_shape=(n_inputs,), activation=activation))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=123))
        for i in range(layers2):
            nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='linear'))
        nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
        return nn

    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasRegressor(build_fn=nn, epochs=epochs, batch_size=batch_size,
                        verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, X, y, cv=kfold, fit_params={'callbacks': [es]}).mean()  # scoring=score_acc
    return score'''
# Create function
def opt_nn(X, y):
    n_inputs = X.shape[1]
    params_nn = {
        'neurons': (5, 50),
        'activation': (0, 9),
        'optimizer': (0, 7),
        'learning_rate': (0.001, 0.2),
        'batch_size': (5, 50),
        'epochs': (20, 100),
        'layers1': (1, 3),
        'layers2': (1, 3),
        'normalization': (0, 1),
        'dropout': (0, 1),
        'dropout_rate': (0, 0.3)}

    def nn_cv(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
        n_inputs = X.shape[1]
        optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
        optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                      'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                      'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                      'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                       'elu', 'exponential', LeakyReLU, 'relu']
        neurons = round(neurons)
        activation = activationL[round(activation)]
        batch_size = round(batch_size)
        epochs = round(epochs)

        def nn():
            opt = Adam(lr=learning_rate)
            nn = Sequential()
            nn.add(Dense(neurons, input_shape=(n_inputs,), activation=activation))
            if normalization > 0.5:
                nn.add(BatchNormalization())
            for i in range(layers1):
                nn.add(Dense(neurons, activation=activation))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate, seed=123))
            for i in range(layers2):
                nn.add(Dense(neurons, activation=activation))
            nn.add(Dense(neurons, activation=activation))
            nn.add(Dense(1, activation='linear'))
            nn.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
            return nn

        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        nn = KerasRegressor(build_fn=nn, epochs=epochs, batch_size=batch_size,
                            verbose=0)
        kfold = KFold(n_splits=5, shuffle=True, random_state=123)
        score = cross_val_score(nn, X, y, cv=kfold, scoring='neg_mean_squared_error',
                                fit_params={'callbacks': [es]}).mean()
        score = np.nan_to_num(score)

        score = score.mean()

        return score

    '''def nn_crossval(params_nn):
        return nn_cv(params_nn, X, y)'''



    nn_bo = BayesianOptimization(nn_cv, params_nn, random_state=111)
    nn_bo.maximize(init_points=50, n_iter=50)

    params_nn_ = nn_bo.max['params']
    learning_rate = params_nn_['learning_rate']
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']
    params_nn_['activation'] = activationL[round(params_nn_['activation'])]
    params_nn_['batch_size'] = round(params_nn_['batch_size'])
    params_nn_['epochs'] = round(params_nn_['epochs'])
    params_nn_['layers1'] = round(params_nn_['layers1'])
    params_nn_['layers2'] = round(params_nn_['layers2'])
    params_nn_['neurons'] = round(params_nn_['neurons'])
    optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'Adam']
    optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                  'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                  'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                  'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
    params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
    params_nn_
    print('param_nn_', params_nn_)


def main():
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
    #X = X.reset_index()
    y = data[out_arr] # , 'CURRENT|2', 'CURRENT|3']]
    #y.fillna(0)


    # CV or split: X_scale, y_scale -> bleiben pd
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    # for shifting
    '''idx_train = np.asarray(y_train.index)

    # step = 1
    # how many points
    window = 1
    # placeholder
    X_plc = X
    # each look back/forward as another feature
    for i in range(window):
        X_shift_bw = X.shift(periods=(i+
                                      1), fill_value=0)
        X_shift_fw = X.shift(periods=-(i+1), fill_value=0)
        X_train_shift_bw = X_shift_bw.loc[idx_train]
        X_train_shift_fw = X_shift_fw.loc[idx_train]
        X_train = pd.concat([X_train, X_train_shift_bw, X_train_shift_fw], axis=1)
        X_plc = pd.concat([X_plc, X_shift_bw, X_shift_fw], axis=1)
    X = X_plc'''
    print('before')
    # kommt NaN vor
    # bei borys muss ich zu 1D vektor machen
    opt_nn(X, y)
    print('after')





if __name__ == '__main__':
        main()