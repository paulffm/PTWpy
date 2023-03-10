# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import xgboost as xgb
from utils import plot_pred, calc_score, data_shift, rolling_stats

show_plot = True

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
    # print(data[measurementVar].str.contains(measurement, na=False).value_counts())
    data = data.loc[data[measurementVar].str.contains(measurement, na=False)]
    print('data after meas', data)

    # output axis
    # 'X', 'Y', 'Z'
    axis = 'Z'

    # insert binning: muss es hier machen und schon die inputs rausschneiden,
    # da ich sonst inkonsistent bekomme mit der L??nge des DF
    data = data[[f'{axis}1_v_dir_lp', f'{axis}1_a_dir_lp', f'{axis}1_FM_lp', f'{axis}1_FB_dir_lp']]
    '''data['BinV'] = pd.qcut(data[f'{axis}1_v_dir_lp'], 4, labels=False)
    data['BinA'] = pd.qcut(data[f'{axis}1_a_dir_lp'], 4, labels=False)'''

    data[f'{axis}1_FR_lp'] = data[f'{axis}1_FM_lp'] - data[f'{axis}1_FB_dir_lp']
    data.fillna(method="ffill")

    # give more input features
    window_sizes = [64, 128, 256]
    data = rolling_stats(data, window_sizes, axis=axis)

    # define input and output
    X = data.drop([f'{axis}1_FR_lp', f'{axis}1_FB_dir_lp', f'{axis}1_FM_lp'], axis=1)
    y = data[f'{axis}1_FR_lp']

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


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    model = xgb.XGBRegressor(learning_rate=params['learning_rate'], subsample=params['subsample'],
                             colsample_bytree=params['colsample_bytree'],
                             colsample_bylevel=params['colsample_bylevel'])

    print('fitting the model')
    model.fit(X_train, y_train.ravel())

    print('predict')
    y_pred = model.predict(X)

    score, idx, y_pred = calc_score(y_pred, y)
    print('Best score:', score)

    # plot score
    if show_plot:
        plot_pred(data, y_pred, y, idx, axis)


if __name__ == '__main__':
    main()
    # X Best score:
    # No forward: 60 back
    '''Number of points with difference > 200: 1 Best score: [200.35260356]'''
