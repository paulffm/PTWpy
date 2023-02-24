# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from random import uniform, randint, choice, choices
from scipy.signal import butter, cheby1, filtfilt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


#### settings ###
# Changes: numerical binning

def plot_pred(data, y_pred, y, idx, axis):
    '''
    :param y_pred:
    :param y_butter:
    :param y:
    :param idx:
    :return:
    '''
    # idx for max diff in plot
    X_diff = (data.iloc[idx])
    X_diff_v = X_diff[f'{axis}1_v_dir_lp']
    X_diff_a = X_diff[f'{axis}1_a_dir_lp']
    y_diff_idx = X_diff.index
    y_mostd = np.asarray(y.iloc[idx])

    # plot of 'normal' current, filtered current, predicted current and all points with diff > 0.1
    # f'{axis}1_v_dir_lp', f'{axis}1_a_dir_lp'
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=data[f'{axis}1_v_dir_lp'], y=y,
                               name=f'{axis}1_FR_lp'))
    fig_v.add_trace(go.Scatter(x=data[f'{axis}1_v_dir_lp'], y=y_pred.flatten(),
                               name=f'{axis}1_FR_lp pred'))
    fig_v.add_trace(go.Scatter(x=X_diff_v, y=y_mostd.flatten(),
                               name=f'most difference', mode='markers',
                               marker=dict(size=10)))
    fig_v.update_layout(
        title=f'{axis}1_FR_lp over v',
        yaxis_title=f'{axis}1_FR_lp',
        xaxis_title=f'{axis}1_v_dir_lp',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig_v.show()

    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=data[f'{axis}1_a_dir_lp'], y=y,
                               name=f'{axis}1_FR_lp'))
    fig_a.add_trace(go.Scatter(x=data[f'{axis}1_a_dir_lp'], y=y_pred.flatten(),
                               name=f'{axis}1_FR_lp pred'))
    fig_a.add_trace(go.Scatter(x=X_diff_a, y=y_mostd.flatten(),
                               name=f'most difference', mode='markers',
                               marker=dict(size=10)))
    fig_a.update_layout(
        title=f'{axis}1_FR_lp over a',
        yaxis_title=f'{axis}1_FR_lp',
        xaxis_title=f'{axis}1_a_dir_lp',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig_a.show()

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=y.index, y=y,
                               name=f'{axis}1_FR_lp'))
    fig_t.add_trace(go.Scatter(x=y.index, y=y_pred.flatten(),
                               name=f'{axis}1_FR_lp pred'))
    fig_t.add_trace(go.Scatter(x=y_diff_idx, y=y_mostd.flatten(),
                               name=f'most difference', mode='markers',
                               marker=dict(size=10)))
    fig_t.update_layout(
        title=f'{axis}1_FR_lp over time',
        yaxis_title=f'{axis}1_FR_lp',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig_t.show()


def scaling_method(method):
    if method == 'MinMax(-1.1)':
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

    elif method == 'MinMax':
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

    elif method == 'Standard':
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

    else:
        raise ValueError

    return scaler_x, scaler_y


def data_shift(X, window, forw, inp='Null'):
    # each look back/forward as another feature
    # falls kein input_namen gegeben, die geshiftet werden sollen => shifte alles
    if inp == 'Null':
        inp = X.columns.values.tolist()

    X_plc = X
    if forw == 1:
        for i in range(window):
            X_shift_bw = X[inp].shift(periods=(i + 1), fill_value=0)
            X_shift_fw = X[inp].shift(periods=-(i + 1), fill_value=0)
            inp_bw = [x + f'_-{i + 1}' for x in inp]
            inp_fw = [x + f'_+{i + 1}' for x in inp]
            X_shift_bw.columns = inp_bw
            X_shift_fw.columns = inp_fw
            X_plc = pd.concat([X_plc, X_shift_bw, X_shift_fw], axis=1)
    else:
        for i in range(window):
            X_shift_bw = X[inp].shift(periods=(i + 1), fill_value=0)
            inp_bw = [x + f'_-{i + 1}' for x in inp]
            X_shift_bw.columns = inp_bw
            X_plc = pd.concat([X_plc, X_shift_bw], axis=1)

    return X_plc


def calc_score(y_pred, y):
    '''
    :param y_pred:
    :param y:
    :param params:
    :return:
    '''

    ''''# inverse transformation
    if params['scaling'] == 1:
        y = params['scaler_y_m'].inverse_transform(y)
        # y = params['scaler_y_r'].inverse_transform(y)

        y_pred = params['scaler_y_m'].inverse_transform(np.asarray(y_pred).reshape(-1, 1))
        # y_pred = params['scaler_y_r'].inverse_transform(y_pred)'''

    # max error
    y_diff = np.abs((np.asarray(y).reshape(-1, 1) - y_pred.reshape(-1, 1)))
    count = np.count_nonzero(y_diff > 200)
    print('Number of points with difference > 200:', count)

    idx = np.argsort(y_diff, axis=0)  # sorts along first axis (down)
    idx = idx[::-1]
    y_diff_sort = np.take_along_axis(y_diff, idx, axis=0)
    score = y_diff_sort[:1]
    # print('Max diff:', score)
    idx = idx[:count].reshape(-1)

    return score.flatten(), idx, y_pred


def main():
    print('Go')

    ## data preparation
    feedVar = '/NC/_N_CH_GD9_ACX/SYG_S9|3'
    measurementVar = 'CH1_ProcessTag2'
    # doesnt exist
    axisVar = '/NC/_N_CH_GD9_ACX/SYG_I9|2'
    loopVar1 = '/NC/_N_CH_GD9_ACX/SYG_I9|3'
    loopVar2 = '/NC/_N_CH_GD9_ACX/SYG_I9|4'
    loopVar3 = '/NC/_N_CH_GD9_ACX/SYG_I9|5'
    msgVar = '/Channel/ProgramInfo/msg|u1'
    blockVar = "/Channel/ProgramInfo/block|u1.2"

    # data: filtering better
    # /Users/paulheller/Desktop/PTW_Data/KohnData/
    file = '2023-01-16T1253_MRM_DMC850_20220509_Filter.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    # print('header', data.columns.values.tolist())
    # print(data)
    data['DateTime'] = pd.to_datetime(data["time_"])
    data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')
    # data = data.sort_values(by="CYCLE")

    machine = "DMC850"
    measurement = "Lineare_Referenzfahrt_Feed"
    # data = data.sort_values(by="CYCLE")
    # data = data.fillna(method="ffill")
    data = data.ffill()
    print('Data before', data)
    data['CH1_ActProgramBlock'] = data['CH1_ActProgramBlock'].fillna(method="ffill")
    print('NaN count', data.isna().sum())
    data.mask(data == 'nan', None).ffill()
    print('NaN count', data.isna().sum())
    print('after nan', data)
    # data = data.loc[data[axisVar] == active_axis]
    # 524.000
    # print(data[measurementVar].str.contains(measurement, na=False).value_counts())
    data = data.loc[data[measurementVar].str.contains(measurement, na=False)]
    print('data after meas', data)

    # output
    # 'X', 'Y', 'Z'
    axis = 'Z'
    # insert binning: muss es hier machen und schon die inputs rausschneiden, da ich sonst inkonsistent bekomme mit der Länge des DF
    data = data[[f'{axis}1_v_dir_lp', f'{axis}1_a_dir_lp', f'{axis}1_FM_lp', f'{axis}1_FB_dir_lp']]
    '''data['BinV'] = pd.qcut(data[f'{axis}1_v_dir_lp'], 4, labels=False)
    data['BinA'] = pd.qcut(data[f'{axis}1_a_dir_lp'], 4, labels=False)'''
    data['MeanV1'] = data[f'{axis}1_v_dir_lp'].rolling(5000).mean()
    data['MeanA1'] = data[f'{axis}1_a_dir_lp'].rolling(5000).mean()
    data['StdV1'] = data[f'{axis}1_v_dir_lp'].rolling(5000).mean()
    data['StdA1'] = data[f'{axis}1_a_dir_lp'].rolling(5000).mean()
    data['MeanV2'] = data[f'{axis}1_v_dir_lp'].rolling(2000).mean()
    data['MeanA2'] = data[f'{axis}1_a_dir_lp'].rolling(2000).mean()
    data['StdV2'] = data[f'{axis}1_v_dir_lp'].rolling(2000).mean()
    data['StdA2'] = data[f'{axis}1_a_dir_lp'].rolling(2000).mean()
    data['MeanV3'] = data[f'{axis}1_v_dir_lp'].rolling(512).mean()
    data['MeanA3'] = data[f'{axis}1_a_dir_lp'].rolling(512).mean()
    data['StdV3'] = data[f'{axis}1_v_dir_lp'].rolling(512).mean()
    data['StdA3'] = data[f'{axis}1_a_dir_lp'].rolling(512).mean()
    data[f'{axis}1_FR_lp'] = data[f'{axis}1_FM_lp'] - data[f'{axis}1_FB_dir_lp']
    data.fillna(method="ffill")
    y = data[f'{axis}1_FR_lp']
    X = data.drop([f'{axis}1_FR_lp', f'{axis}1_FB_dir_lp', f'{axis}1_FM_lp'], axis=1)
    print(X)
    print(y, y.shape)
    params = {'max_depth': 11,
              'learning_rate': 0.06,
              'subsample': 0.7,
              'colsample_bytree': 0.6,
              'colsample_bylevel': 0.9,
              'shifting': 1,
              'scaling': 0,
              'step_size': 75,  # 32
              'forward': 0}  # 1

    if params['shifting'] == 1:
        window = params['step_size']
        forw = params['forward']
        inp_shift = [f'{axis}1_v_dir_lp', f'{axis}1_a_dir_lp']
        X = data_shift(X, window, forw, inp_shift)
    print('X after shift', data)  # , X[f'{axis}1_FR_lp', f'{axis}1_FB_dir_lp', f'{axis}1_FM_lp'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    model = xgb.XGBRegressor(learning_rate=params['learning_rate'], subsample=params['subsample'],
                             colsample_bytree=params['colsample_bytree'],
                             colsample_bylevel=params['colsample_bylevel'])
    print('X_train', X_train)
    print('fitting the model')
    model.fit(X_train, y_train.ravel())

    print('predict')
    y_pred = model.predict(X)
    print('y_pred', y_pred, y_pred.shape)

    score, idx, y_pred = calc_score(y_pred, y)
    print('Best score:', score)

    # plot score
    plot_pred(data, y_pred, y, idx, axis)


if __name__ == '__main__':
    main()
    # X Best score:
    # No forward: 60 back
    '''Number of points with difference > 200: 1 Best score: [200.35260356]'''
