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


def data_shift(X, window, forw, inp=0):

    # each look back/forward as another feature
    # falls kein input_namen gegeben, die geshiftet werden sollen => shifte alles
    if inp == 0:
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


class random_search:
    def __init__(self, n_iter, axis):
        self.n_iter = n_iter
        self.best_params = dict()
        self.best_score = 1e9
        self.axis = axis



    def __build_model(self, model):
        '''
        :param model:
        :return:
        '''
        if model == 'RandomForrest':
            pass
        elif model == 'XGBoost':

            params = {'max_depth': randint(3, 20),
                           'learning_rate': choice([0.025, 0.0275, 0.03, 0.035, 0.04, 0.05]),
                           'subsample': choice([0.3, 0.4, 0.5, 0.6, 0.7]),
                           'colsample_bytree': choice([0.3, 0.4, 0.5, 0.6]),
                           'colsample_bylevel': choice([0.3, 0.5, 0.6, 0.7, 0.9, 1])}

            rf = xgb.XGBRegressor(learning_rate=params['learning_rate'], subsample=params['subsample'],
                                  colsample_bytree=params['colsample_bytree'],
                                  colsample_bylevel=params['colsample_bylevel'])
            return rf, params
        else:
            pass


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
            #y = params['scaler_y_r'].inverse_transform(y)

            y_pred = params['scaler_y_m'].inverse_transform(np.asarray(y_pred).reshape(-1, 1))
            #y_pred = params['scaler_y_r'].inverse_transform(y_pred)

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

        return score.flatten(), idx, y_pred, y

    def fit_predict(self, model, data, y_all):
        '''
        :param data:
        :param y:
        :return:
        '''

        best_ypred = []
        best_idx = []
        for n_i in range(self.n_iter):
            y = y_all
            print(f'Iteration: {n_i + 1}')

            # build model
            rf, params = self.__build_model(model)

            # params as scipy object: and rest
            params['shifting'] = randint(0, 1)
            params['scaling'] = 0 #randint(0, 1)

            inp = [f'{self.axis}1_v_dir_lp', f'{self.axis}1_a_dir_lp']
            # inp.append(params['inputs'])
            X = data[inp]

            if params['shifting'] == 1:
                window = randint(1, 20)
                params['step_size'] = window
                forw = randint(0, 1)
                params['forward'] = forw
                X = data_shift(X, window, forw, inp)


            if params['scaling'] == 1:
                method = choice(['Standard', 'MinMax', 'MinMax(-1.1)'])
                params['scaling_method'] = method

                '''scaler_x_r = RobustScaler()
                scaler_y_r = RobustScaler()
                X = scaler_x_r.fit_transform(X)
                y = scaler_y_r.fit_transform(np.asarray(y).reshape(-1, 1))
                params['scaler_y_r'] = scaler_y_r'''

                scaler_x_m, scaler_y_m = scaling_method(method)
                X = scaler_x_m.fit_transform(X)
                y = scaler_y_m.fit_transform(np.asarray(y).reshape(-1, 1))
                params['scaler_y_m'] = scaler_y_m

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
            rf.fit(X_train, y_train.ravel())
            y_pred = rf.predict(X)

            score, idx, y_pred, y = self.__calc_score(np.asarray(y_pred).reshape(-1, 1), np.asarray(y).reshape(-1, 1),
                                                      params)

            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                best_ypred = y_pred
                best_idx = idx
                print('Best Params:', self.best_params)

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
    #/Users/paulheller/Desktop/PTW_Data/KohnData/
    file = '2023-01-16T1253_MRM_DMC850_20220509_Filter.csv'
    data = pd.read_csv(file, sep=',', header=0, index_col=0, parse_dates=True, decimal=".")
    print('header', data.columns.values.tolist())
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
    # 'X', 'Y', 'Z'
    axis = 'X'
    data[f'{axis}1_FR_lp'] = data[f'{axis}1_FM_lp'] - data[f'{axis}1_FB_dir_lp']
    y = data[f'{axis}1_FR_lp']
    data = data.drop([f'{axis}1_FR_lp'], axis=1)

    # insert binning
    data['BinV'] = pd.qcut(data[f'{axis}1_v_dir_lp'], 4, labels=False)
    data['BinA'] = pd.qcut(data[f'{axis}1_a_dir_lp'], 4, labels=False)
    data['MeanV'] = data[f'{axis}1_v_dir_lp'].rolling(5).mean()
    data['MeanA'] = data[f'{axis}1_a_dir_lp'].rolling(5).mean()
    data['StdV'] = data[f'{axis}1_v_dir_lp'].rolling(5).mean()
    data['StdA'] = data[f'{axis}1_a_dir_lp'].rolling(5).mean()
    data = data.dropna()
    print(data)


    # plot score
    #y_pred = np.loadtxt(f'/Users/paulheller/Desktop/PTW_Data/KohnData/X/y_pred_{axis}.csv', delimiter=',')
    #idx = np.loadtxt(f'/Users/paulheller/Desktop/PTW_Data/KohnData/X/idx_{axis}.csv', delimiter=',').astype(int)
    #print(idx, idx.shape)
    #print(y_pred, y_pred.shape)
    #plot_pred(data, y_pred, y_butter, y, idx, axis)

    n_iter = 1
    # XGBoost,RandomForrest
    model = 'XGBoost'
    # start random search:
    search_rg = random_search(n_iter, axis)
    y_pred, idx = search_rg.fit_predict(model, data, y)
    print('Best params', search_rg.best_params)
    print('Best score', search_rg.best_score)

    # plot score
    plot_pred(data, y_pred, y, idx, axis)


if __name__ == '__main__':
    main()
# RF after 21 iterations
'''Best Params: {'shifting': 1, 'scaling': 1, 'step_size': 4, 'forward': 0, 'scaling_method': 'Standard', 'scaler_y_r': RobustScaler(), 'scaler_y_m': StandardScaler()}
Score: [474.39711064]; Best Score: [474.39711064]'''

# XGB:
# X1
'''
Number of points with difference > 200: 5379
Best Params: {'shifting': 1, 'scaling': 0, 'step_size': 16, 'forward': 1}
Score: [263.54339314]; Best Score: [263.54339314]'''
'''Best params {'max_depth': 16, 'learning_rate': 0.025, 'subsample': 0.5, 'colsample_bytree': 0.4, 'colsample_bylevel': 1, 'shifting': 1, 'scaling': 1, 'step_size': 9, 'forward': 1, 'scaling_method': 'MinMax(-1.1)', 'scaler_y_r': RobustScaler(), 'scaler_y_m': MinMaxScaler(feature_range=(-1, 1))}
Best score [400.490167]'''

'''Number of points with difference > 200: 668
Best Params: {'max_depth': 17, 'learning_rate': 0.03, 'subsample': 0.4, 'colsample_bytree': 0.4, 'colsample_bylevel': 0.6, 'shifting': 1, 'scaling': 1, 'step_size': 20, 'forward': 0, 'scaling_method': 'MinMax(-1.1)', 'scaler_y_r': RobustScaler(), 'scaler_y_m': MinMaxScaler(feature_range=(-1, 1))}
Score: [427.7763533]; Best Score: [427.7763533]'''
# Y1
'''Number of points with difference > 200: 237
Best Params: {'shifting': 1, 'scaling': 0, 'step_size': 17, 'forward': 1}
Score: [250.26284924]; Best Score: [250.26284924]'''
# Z1
'''Number of points with difference > 200: 912
Best Params: {'shifting': 1, 'scaling': 0, 'step_size': 16, 'forward': 1}
Score: [263.54339314]; Best Score: [263.54339314]'''