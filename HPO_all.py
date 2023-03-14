# Import packages
import numpy as np
import pandas as pd
import xgboost as xgb
from utils import plot_pred,  rolling_stats
from RandomSearch import RandomSearch
from scipy.stats import uniform, randint

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
    file = '/Users/paulheller/Desktop/PTW_Data/KohnData/2023-01-16T1253_MRM_DMC850_20220509_Filter.csv'
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


    # output axis
    # 'X', 'Y', 'Z'
    axis = 'Z'

    # insert binning: muss es hier machen und schon die inputs rausschneiden,
    # da ich sonst inkonsistent bekomme mit der Länge des DF
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

    # Random Search
    n_iter = 150
    # XGBoost,RandomForrest
    model = 'XGBoost'

    # muss leider als liste noch definiert sein
    param_distribution = {
        'max_depth': list(range(6, 15)),
        'learning_rate': [0.050, 0.055, 0.0575, 0.06, 0.065, 0.07, 0.075, 0.8],
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, ],
        'min_child_weight': list(range(1, 2)),
        'shifting': list(range(0, 1)),
        'step_size': list(range(20, 80)),
        'forward': list(range(0, 1)),
        'scaling': list(range(0, 1)),
        'scaling_method': ['MinMax', 'MinMax(-1.1)', 'Standard']
    }

    search_rg = RandomSearch(param_distribution, n_iter, axis)

    y_pred, idx = search_rg.fit_predict(data, y)
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

    '''    y_diff = np.abs((np.asarray(y).reshape(-1, 1) - y_pred.reshape(-1, 1)))
ValueError: operands could not be broadcast together with shapes (525454,1) (210182,1) '''