# Import packages
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint

def data_shift(X, X_train, window):
    X_plc = X
    idx_train = np.asarray(X_train.index)
    # each look back/forward as another feature
    for i in range(window):
        X_shift_bw = X.shift(periods=(i + 1), fill_value=0)
        X_shift_fw = X.shift(periods=-(i + 1), fill_value=0)
        X_train_shift_bw = X_shift_bw.loc[idx_train]
        X_train_shift_fw = X_shift_fw.loc[idx_train]
        X_train = pd.concat([X_train, X_train_shift_bw, X_train_shift_fw], axis=1)
        X_plc = pd.concat([X_plc, X_shift_bw, X_shift_fw], axis=1)

    return X_plc, X_train

def main():
    # data preparation
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
    # schlechter, wenn diese beiden nicht berücksichtigt werden
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]

    # Parameters to choose:
    # input and output

    inp = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}']
    out_arr = [f'CURRENT|{active_axis}']






    X = data[inp]
    y = data[out_arr]
    ## data preparation over

    # CV or split: X_scale, y_scale -> bleiben pd
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # shifting forward and backward
    window = 1
    X, X_train = data_shift(X, X_train, window)

    # rf grid
    rf_params_grid = {"bootstrap": [True],
                 "max_depth": [6, 8, 10, 12, 14],
                 "max_features": ['auto', 'sqrt', 'log2'],
                 "min_samples_leaf": [2, 3, 4],
                 "min_samples_split": [2, 3, 4, 5],
                 "n_estimators": [100, 200, 300],
                 "criterion": ['squared_error', 'absolute_error']}

    rf_params_rand = {"bootstrap": [True, False],
                      "max_depth": randint(4, 14),
                      "max_features": ['auto', 'sqrt', 'log2'],
                      "min_samples_leaf": randint(2, 6),
                      "min_samples_split": randint(2, 6),
                      "n_estimators": randint(100, 300),
                      "criterion": ['squared_error', 'absolute_error']}

    rf_params_bs = {"bootstrap": Categorical([True, False]),
                    "max_depth": Integer(4, 14),
                    "max_features": Categorical(['auto', 'sqrt', 'log2']),
                    "min_samples_leaf": Integer(2, 6),
                    "min_samples_split": Integer(2, 6),
                    "n_estimators": Integer(100, 300),
                    "criterion": Categorical(['squared_error', 'absolute_error'])}

    # for XGBoost:
    xgb_params_rand = {'estimator__max_depth': randint(3, 18),
                       'estimator__gamma': randint(1, 9),
                       'estimator__reg_alpha': randint(40, 180),
                       'estimator__reg_lambda': uniform(0, 1),
                       'estimator__colsample_bytree': uniform(0.5, 1),
                       'estimator__min_child_weight': randint(0, 10),
                       'estimator__n_estimators': randint(100, 1000),
                       'estimator__learning_rate': loguniform(0.001, 0.1)}

    xgb_params_bs = {'estimator__max_depth': Integer(3, 18),
                     'estimator__gamma': Integer(1, 9),
                     'estimator__reg_alpha': Integer(40, 180),
                     'estimator__reg_lambda': Real(0, 1),
                     'estimator__colsample_bytree': Real(0.5, 1),
                     'estimator__min_child_weight': Integer(0, 10),
                     'estimator__learning_rate': Real(0.005, 0.1),
                     'estimator__n_estimators': Integer(100, 1000)}

    print('build model')

    n_iter_search = 2


    # optimize
    print('opt')
    # # RF
    rf_reg = RandomForestRegressor()
    #search_reg = GridSearchCV(rf_reg, rf_params_grid, cv=3, n_iter=n_iter_search, scoring='max_error')
    #search_reg = RandomizedSearchCV(rf_reg, rf_params_rand, cv=5, n_iter=n_iter_search, scoring='max_error')
    search_reg = BayesSearchCV(rf_reg, rf_params_bs, cv=5, scoring='max_error', n_iter=n_iter_search)

    search_reg.fit(X_train, np.ravel(y_train))
    print(search_reg.best_params_)
    print(f'max error: {search_reg.best_score_}')
    # test prediction
    y_pred = search_reg.predict(X)

    # error values
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print('MSE:', mse)
    print('MAE:', mae)

    # maximal difference: should be <0.1
    y_diff_sort = (y - np.reshape(y_pred, (y_pred.shape[0], 1))).abs().sort_values(by=[f'CURRENT|{active_axis}'],
                                                                                   ascending=False)
    k = 4
    y_diff_k = y_diff_sort.iloc[:k]
    print('max diff', y_diff_k.iloc[0])
    print(f'{k} max diff: {y_diff_k}')


if __name__ == '__main__':
        main()

        # to do:
        # why nan in grids, randoms with NN, RF testen?
        # skopt BayesianCV testen -> ähnlich wie grid, random?
        # gp minimize or hp bandster
        #
        # why bayesopt NaN
