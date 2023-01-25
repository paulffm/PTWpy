import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skopt.utils import use_named_args
import itertools
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from skopt.callbacks import EarlyStopper
def conv_list2dict(d):
    it = iter(d)
    res_dct = dict(zip(it, it))
    return res_dct

def splitDict(d, n):
    i = iter(d.items())
    d1 = dict(itertools.islice(i, n))
    d2 = dict(i)
    return d1, d2

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



def main():
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

    active_axis = 2
    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]
    ## data preparation over


    inp = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}']
    inp_all_axis = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}',
                    f'DES_POS|2', f'VEL_FFW|2', f'TORQUE_FFW|2',
                    f'DES_POS|3', f'VEL_FFW|3', f'TORQUE_FFW|3']
    input_list = [inp, inp_all_axis]

    # scaling: yes=1 no=0 ->0 no scaling is the best
    # Scaling method: 'MinMax(0.1)', 'MinMax()', 'Standard'
    scaling = 0
    method = 'Standard'

    # shifting: yes=1 no=0, window_size: 3 Benchmark, 5 better, 10 worse ->1 makes max difference better
    # forw: if forward and backward shifting or only backwards: both=1 only backward=0
    shifting = 1
    window = 3
    forw = 1


    # output
    out_arr = [f'CURRENT|{active_axis}']


    space = [Categorical([True, False], name="bootstrap"),
                    Integer(4, 14, name="max_depth"),
                    Categorical(['auto', 'sqrt', 'log2'], name="max_features"),
                    Integer(2, 6, name="min_samples_leaf"),
                    Integer(2, 6, name="min_samples_split"),
                    Integer(100, 300, name="n_estimators"),
                    Categorical(['squared_error', 'absolute_error'], name="criterion"),
                    Integer(0, 1, name='input'),
                    Integer(0, 1, name='shifting')]
    print(space[0])

    space_d = {"bootstrap": Categorical([True, False]),
                    "max_depth": Integer(4, 14),
                    "max_features": Categorical(['auto', 'sqrt', 'log2']),
                    "min_samples_leaf": Integer(2, 6),
                    "min_samples_split": Integer(2, 6),
                    "n_estimators": Integer(100, 300),
                    "criterion": Categorical(['squared_error', 'absolute_error']),
                    'input': Integer(0, 1),
                    'shifting': Integer(0, 1)
               }
    print(conv_list2dict(space))

    ### Parameter to choose:

    def objective(space):

        space_rf = space[:-2]
        space_po = space[-2:]
        print(space_rf)
        space_rf_dct = conv_list2dict(space_rf)
        #space_rf, space_po = splitDict(space, 7)

        reg = RandomForestRegressor()
        reg.get_params()
        reg.set_params(**space_rf_dct)
        # space_po[0]
        X = data[input_list[space_po[0]]]
        y = data[out_arr]

        # CV or split: X_scale, y_scale -> bleiben pd
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

        # Scaling: hier erst, sonst probleme mit Index
        if scaling == 1:
            scaler_x, scaler_y, scaler_x_tr, scaler_y_tr = scaling(method)

            scaler_x = scaler_x.fit(X)
            scaler_y = scaler_y.fit(y)

            X = scaler_x.transform(X)
            y = scaler_y.transform(y)

            # scaling train
            scaler_x_tr = scaler_x_tr.fit(X_train)
            scaler_y_tr = scaler_y_tr.fit(y_train)

            X_train = scaler_x_tr.transform(X_train)
            y_train = scaler_y_tr.transform(y_train)

        # shifting forward and backward
        #space_po[1]
        if space_po[1] == 1:
            X, X_train = data_shift(X, X_train, window, forw)

        return -np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1,
                                        scoring="neg_mean_absolute_error"))

    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

    print('params')
    for i in range(10):
        print(res_gp.x[i])



    # inverse scaling
    '''if scaling == 1:

        X_train = scaler_x_tr.inverse_transform(X_train)
        X = scaler_x.inverse_transform(X)
        y = scaler_y.inverse_transform(y)
        y_pred = scaler_y.inverse_transform(y_pred)
        y_pred = y_pred.reshape(-1, 1)
        # max error
        y_diff = np.sort(np.abs(np.asarray(y) - np.asarray(y_pred)), axis=0)
        y_max = y_diff[:-1]
        print('sorted', y_max)
        print('max diff', y_max[0])'''


if __name__ == '__main__':
        main()
