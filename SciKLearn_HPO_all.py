# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from random import uniform, randint, choice, choices
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from utils import plot_pred, calc_score, data_shift, scaling_method, rolling_stats

show_plot = True
# params = {k: dist.rvs() for k, dist in param_distributions.items()}
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

            params = {'max_depth': randint(6, 15),
                      'learning_rate': choice([0.050, 0.055, 0.0575, 0.06, 0.065, 0.07, 0.075, 0.8]),
                      'subsample': choice([0.4, 0.5, 0.6, 0.7, 0.8]),
                      'colsample_bytree': choice([0.5, 0.6, 0.7, 0.8, 0.9]),
                      'colsample_bylevel': choice([0.4, 0.5, 0.6, 0.7, ]),
                      'min_child_weight': choice([1, 2, 3, 4])
                      }

            rf = xgb.XGBRegressor(learning_rate=params['learning_rate'], subsample=params['subsample'],
                                  colsample_bytree=params['colsample_bytree'],
                                  colsample_bylevel=params['colsample_bylevel'])
            return rf, params
        else:
            pass

    def fit_predict(self, model, data, y_all):
        '''
        :param data:
        :param y:
        :return:
        '''

        best_ypred = []
        best_idx = []
        for n_i in range(self.n_iter):

            # otheriwse bug with scaling
            y = y_all
            print(f'Iteration: {n_i + 1}')

            # build model (better would be a method that I import)
            rf, params = self.__build_model(model)

            # To DO:params as scipy object: and rest
            params['shifting'] = 1  # randint(0, 1)
            params['scaling'] = 0  # randint(0, 1)

            inp_shift = [f'{self.axis}1_v_dir_lp', f'{self.axis}1_a_dir_lp']
            # inp.append(params['inputs'])
            # X = data[inp]
            X = data

            if params['shifting'] == 1:
                window = randint(19, 40)
                params['step_size'] = window
                forw = randint(0, 1)
                params['forward'] = forw
                X = data_shift(X, window, forw, inp_shift)

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

            score, idx, y_pred, y = calc_score(np.asarray(y_pred).reshape(-1, 1), np.asarray(y).reshape(-1, 1),
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
    # /Users/paulheller/Desktop/PTW_Data/KohnData/
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

    # output axis
    # 'X', 'Y', 'Z'
    axis = 'Z'

    # insert binning: muss es hier machen und schon die inputs rausschneiden,
    # da ich sonst inkonsistent bekomme mit der Länge des DF
    data = data[[f'{axis}1_v_dir_lp', f'{axis}1_a_dir_lp', f'{axis}1_FM_lp', f'{axis}1_FB_dir_lp']]
    data['BinV'] = pd.qcut(data[f'{axis}1_v_dir_lp'], 4, labels=False)
    data['BinA'] = pd.qcut(data[f'{axis}1_a_dir_lp'], 4, labels=False)

    data[f'{axis}1_FR_lp'] = data[f'{axis}1_FM_lp'] - data[f'{axis}1_FB_dir_lp']
    data = data.fillna(method="ffill")
    y = data[f'{axis}1_FR_lp']
    data = data.drop([f'{axis}1_FR_lp', f'{axis}1_FB_dir_lp', f'{axis}1_FM_lp'], axis=1)

    # Random Search
    n_iter = 150
    # XGBoost,RandomForrest
    model = 'XGBoost'

    search_rg = random_search(n_iter, axis)

    y_pred, idx = search_rg.fit_predict(model, data, y)
    np.savetxt(f'y_pred_XGB_{axis}.csv', y_pred, delimiter=',')
    np.savetxt(f'idx_XGB_{axis}.csv', idx, delimiter=',')

    # also save and import dict
    print('Best params', search_rg.best_params)
    print('Best score', search_rg.best_score)

    # plot score
    if show_plot:
        plot_pred(data, y_pred, y, idx, axis)


if __name__ == '__main__':
    main()
# Z
'''Best params {'max_depth': 12, 'learning_rate': 0.075, 'subsample': 0.5, 'colsample_bytree': 0.5, 
'colsample_bylevel': 0.7, 'min_child_weight': 1, 'shifting': 1, 'scaling': 0, 'step_size': 23, 'forward': 1}
Best score [210.17172709]'''

# X
'''"Best Params: {'max_depth': 11, 'learning_rate': 0.06, 'subsample': 0.7, 
'colsample_bytree': 0.6, 'colsample_bylevel': 0.9, 'shifting': 1, 'scaling': 0, 
'step_size': 32, 'forward': 1}\nScore: [177.19018434]; Best Score: [177.19018434]"'''