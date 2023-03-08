# Import packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

def plot_pred(data, y_pred, y, idx, axis):
    '''
    :param data:
    :param y_pred:
    :param y:
    :param idx:
    :param axis:
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
    '''
    :param method:
    :return:
    '''

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
    '''
    :param X:
    :param window:
    :param forw:
    :param inp:
    :return:
    '''

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


def calc_score(y_pred, y, params: dict, threshold=200):

    '''
    :param y_pred:
    :param y:
    :param threshold:
    :return:
    '''

    # inverse transformation
    if params['scaling'] == 1:
        y = params['scaler_y_m'].inverse_transform(y)
        y_pred = params['scaler_y_m'].inverse_transform(np.asarray(y_pred).reshape(-1, 1))

    # max error
    y_diff = np.abs((np.asarray(y).reshape(-1, 1) - y_pred.reshape(-1, 1)))
    count = np.count_nonzero(y_diff > threshold)
    print(f'Number of points with difference > {threshold}:', count)

    idx = np.argsort(y_diff, axis=0)  # sorts along first axis (down)
    idx = idx[::-1]
    y_diff_sort = np.take_along_axis(y_diff, idx, axis=0)
    score = y_diff_sort[:1]
    # print('Max diff:', score)
    idx = idx[:count].reshape(-1)

    return score.flatten(), idx, y_pred

def rolling_stats(data: pd.Dataframe, window_sizes: list, axis):
    '''
    :param data:
    :param window_sizes:
    :param axis:
    :return:
    '''

    for w_i in (window_sizes):
        data[f'MeanV{w_i}'] = data[f'{axis}1_v_dir_lp'].rolling(w_i).mean()
        data[f'MeanA{w_i}'] = data[f'{axis}1_a_dir_lp'].rolling(w_i).mean()
        data[f'StdV{w_i}'] = data[f'{axis}1_v_dir_lp'].rolling(w_i).std()
        data[f'StdA1{w_i}'] = data[f'{axis}1_a_dir_lp'].rolling(w_i).std()

    return data

def evaluate_model(X, y, params):
    # Create a logistic regression model with the given parameters
    model = xgb.XGBRegressor(learning_rate=params['learning_rate'], subsample=params['subsample'],
                                  colsample_bytree=params['colsample_bytree'],
                                  colsample_bylevel=params['colsample_bylevel'])


    # Train the model on the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model.predict(X_test)
    score, idx, y_pred = calc_score(np.asarray(y_pred).reshape(-1, 1), np.asarray(y).reshape(-1, 1), params)

    return score, idx, y_pred



