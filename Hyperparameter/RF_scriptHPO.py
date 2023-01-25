import numpy as np
import pandas as pd
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
from scipy.stats import uniform, randint

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

def scaling_method(method):
    if method == 'MinMax(0.1)':
        scaler_x_test = MinMaxScaler(feature_range=(0, 1))
        scaler_x_tr = MinMaxScaler(feature_range=(0, 1))
        scaler_y_test = MinMaxScaler(feature_range=(0, 1))
        scaler_y_tr= MinMaxScaler(feature_range=(0, 1))
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

    # input: DES_POS, VEL_FFW, TORQUE_FFW: 0=from active axis, 1=from all axis ->0
    input = 0

    # scaling: yes=1 no=0 ->no
    # Scaling method: 'MinMax(0.1)', 'MinMax()', 'Standard'
    scaling = 0
    method = 'Standard'

    # shifting: yes=1 no=0, window_size: -> yes:, step=3, forw=1
    # forw: if forward and backward shifting or only backwards: both=1 only backward=0
    shifting = 1
    step = 3
    forw = 1

    # which search method we use: 0=GridSearch, 1=RandomSearch, 2=BayesSearch
    # BayesSearch or RandomSearch prefered
    search_method = 1

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

    # data:
    file = r'/Users/paulheller/PycharmProjects/PTWPy/PTWKohn/Data/2022-07-28_MRM_DMC850_20220509.MPF.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    #print('header', data.columns.values.tolist())
    data['DateTime'] = pd.to_datetime(data["_time"])
    data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(by="CYCLE")

    # filtering better
    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    data = data.sort_values(by="CYCLE")
    data = data.fillna(method="ffill")
    data = data.loc[data[axisVar] == active_axis]
    data = data.loc[data[measurementVar].str.contains(measurement)]


    out_arr = [f'CURRENT|{active_axis}']
    X = data[input_list[input]]
    y = data[out_arr]

    # CV or split: X_scale, y_scale -> bleiben pd
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # shifting forward and backward:
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


    # rf parameter to test
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

    # optimize
    print('opt') # scoring= which one to choose?
    # RF
    n_iter_search = 10
    rf_reg = RandomForestRegressor()
    if search_method == 0:
        search_reg = GridSearchCV(rf_reg, rf_params_grid, cv=5, scoring='accuracy')
    elif search_method == 1:
        search_reg = RandomizedSearchCV(rf_reg, rf_params_rand, cv=5, scoring='max_error', n_iter=n_iter_search)
    elif search_method == 2:
        search_reg = BayesSearchCV(rf_reg, rf_params_bs, cv=5, scoring='max_error', n_iter=n_iter_search)

    search_reg.fit(X_train, np.ravel(y_train))
    print(search_reg.best_params_)
    print(f' best score: {search_reg.best_score_}')
    # test prediction
    y_pred = search_reg.predict(X)

    # inverse scaling
    if scaling == 1:

        X_train = scaler_x_tr.inverse_transform(X_train)
        X = scaler_x.inverse_transform(X)
        y = scaler_y.inverse_transform(y)
        y_pred = scaler_y.inverse_transform(y_pred)
        y_pred = y_pred.reshape(-1, 1)
        # max error
        y_diff = np.sort(np.abs(np.asarray(y) - np.asarray(y_pred)), axis=0)
        y_max = y_diff[::-1]
        k = 4
        print('max diff', y_max[:k])

    elif scaling == 0:

        # without scaling I can keep my index with this type of method to compute max diff
        y_diff_sort = (y - np.reshape(y_pred, (y_pred.shape[0], 1))).abs().sort_values(by=[f'CURRENT|{active_axis}'],
                                                                                       ascending=False)
        k = 4
        y_diff_k = y_diff_sort.iloc[:k]
        print(f'{k} max diff: {y_diff_k}')

    # error values
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print('MSE:', mse)
    print('MAE:', mae)



if __name__ == '__main__':
        main()
