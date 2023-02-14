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


def plot_pred(y_pred, y_butter, y, idx):
    '''
    :param y_pred:
    :param y_butter:
    :param y:
    :param idx:
    :return:
    '''
    # idx for max diff in plot
    y_diff_idx = (y.iloc[idx]).index
    y_mostd = y_butter[idx]

    # plot of 'normal' current, filtered current, predicted current and all points with diff > 0.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y,
                             name=f'X1_FR_lp'))
    fig.add_trace(go.Scatter(x=y.index, y=y_pred,
                             name='X1_FR_lp pred'))
    fig.add_trace(go.Scatter(x=y.index, y=y_butter.flatten(),
                             name=f'X1_FR_lp filtered'))
    fig.add_trace(go.Scatter(x=y_diff_idx, y=y_mostd.flatten(),
                             name=f'most difference', mode='markers',
                             marker=dict(size=10)))

    fig.update_layout(
        title=f'X1_FR_lp over time',
        yaxis_title=f'X1_FR_lp',
        xaxis_title=f'time',
        font=dict(family="Tahoma", size=18, color="Black"))
    fig.show()


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


def data_shift(X, window, forw):
    # each look back/forward as another feature
    inp = X.columns.values.tolist()
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


class random_search:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.best_params = dict()
        self.best_score = 1e9

    def __calc_score(self, y_pred, y, params):
        '''
        :param y_pred:
        :param y:
        :param params:
        :return:
        '''

        # inverse transformation
        if params['scaling'] == 1:
            y = params['scaler_y_m'].inverse_transform(y)
            y = params['scaler_y_r'].inverse_transform(y)

            y_pred = params['scaler_y_m'].inverse_transform(np.asarray(y_pred).reshape(-1, 1))
            y_pred = params['scaler_y_r'].inverse_transform(y_pred)

        # max error
        y_diff = np.abs((y) - (y_pred.reshape(-1, 1)))
        count = np.count_nonzero(y_diff > 200)
        print('Number of points with difference > 200:', count)

        idx = np.argsort(y_diff, axis=0)  # sorts along first axis (down)
        idx = idx[::-1]
        y_diff_sort = np.take_along_axis(y_diff, idx, axis=0)
        score = y_diff_sort[:1]
        # print('Max diff:', score)
        idx = idx[:count].reshape(-1)

        return score.flatten(), idx, y_pred

    def fit_predict(self, data, y):
        '''
        :param data:
        :param y:
        :return:
        '''

        best_ypred = []
        best_idx = []
        for n_i in range(self.n_iter):
            y_i = y
            print(f'Iteration: {n_i + 1}')

            rf = RandomForestRegressor(n_estimators=100)

            params = {
                # 'inputs': choices(0,1),
                'shifting': randint(0, 1),
                'scaling': randint(0, 1)

            }
            inp = ['X1_v_dir_lp', 'X1_a_dir_lp']
            # inp.append(params['inputs'])
            X = data[inp]

            if params['shifting'] == 1:
                window = randint(1, 4)
                params['step_size'] = window
                forw = randint(0, 1)
                params['forward'] = forw
                X = data_shift(X, window, forw)

            if params['scaling'] == 1:
                method = choice(['Standard', 'MinMax', 'MinMax(-1.1)'])
                params['scaling_method'] = method

                scaler_x_r = RobustScaler()
                scaler_y_r = RobustScaler()
                X = scaler_x_r.fit_transform(X)
                y_i = scaler_y_r.fit_transform(np.asarray(y_i).reshape(-1, 1))
                params['scaler_y_r'] = scaler_y_r

                scaler_x_m, scaler_y_m = scaling_method(method)
                X = scaler_x_m.fit_transform(X)
                y_i = scaler_y_m.fit_transform(np.asarray(y_i).reshape(-1, 1))
                params['scaler_y_m'] = scaler_y_m

            X_train, X_test, y_train, y_test = train_test_split(X, y_i, test_size=0.4, random_state=1)
            rf.fit(X_train, y_train.ravel())
            y_pred = rf.predict(X)

            score, idx, y_pred = self.__calc_score(np.asarray(y_pred).reshape(-1, 1), np.asarray(y_i).reshape(-1, 1),
                                                        params)

            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                best_ypred = y_pred
                best_idx = idx
                print('Best Params:', self.best_params)
            # print('best y_pred', best_ypred)
            # print('y', y)
            print(f'Score: {score}; Best Score: {self.best_score}')

        return best_ypred, best_idx


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
    data = data.fillna(method="ffill")
    # print('NaN count', data.isna().sum())
    data.mask(data == 'nan', None).ffill()
    # data = data.loc[data[axisVar] == active_axis]
    # print(data[measurementVar].str.contains(measurement, na=False).value_counts())
    data = data.loc[data[measurementVar].str.contains(measurement, na=False)]

    # output
    data['X1_FR_lp'] = data['X1_FM_lp'] - data['X1_FB_dir_lp']
    y = data['X1_FR_lp']
    # print(data)
    data = data.drop(['X1_FR_lp'], axis=1)
    print(data)

    # filter signals
    order = 1
    b, a = butter(order, Wn=0.1, btype='lowpass')
    y_butter = filtfilt(b, a, y, axis=0)
    print('ybutter', y_butter)

    n_iter = 25
    # start random search:
    search_rg = random_search(n_iter)
    y_pred, idx = search_rg.fit_predict(data, y_butter)
    print('Best params', search_rg.best_params)
    print('Best score', search_rg.best_score)

    # plot score
    plot_pred(y_pred, y_butter, y, idx)


if __name__ == '__main__':
    main()