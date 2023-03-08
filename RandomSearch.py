# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from random import uniform, randint, choice, choices
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from utils import plot_pred, calc_score, data_shift, scaling_method, rolling_stats, evaluate_model
import random


class RandomSearch:
    def __init__(self, param_distributions, n_iter, axis):
        self.n_iter = n_iter
        self.param_distributions = param_distributions
        self.axis = axis


    def fit_predict(self, X, y):
        best_score = -np.inf
        best_params = dict()
        best_ypred = []
        best_idx = []

        for i in range(self.n_iter):
            # Sample a set of parameters from the distributions
            #params = {k: dist.rvs() for k, dist in self.param_distributions.items()}

            params = {k: random.choice(dist) for k, dist in self.param_distributions.items()}

            if params['shifting'] == 1:
                inp_shift = [f'{self.axis}1_v_dir_lp', f'{self.axis}1_a_dir_lp']
                X = data_shift(X, params['step_size'], params['forward'], inp_shift)

            if params['scaling'] == 1:

                scaler_x_m, scaler_y_m = scaling_method(params['scaling_method'])
                X = scaler_x_m.fit_transform(X)
                y = scaler_y_m.fit_transform(np.asarray(y).reshape(-1, 1))
                params['scaler_y_m'] = scaler_y_m

            # Evaluate the performance of the model with the current set of parameters
            score, idx, y_pred = evaluate_model(X, y, params)

            # If the current score is better than the best score, update the best score and parameters
            if score < best_score:
                best_score = score
                best_params = params
                best_idx = idx
                best_ypred = y_pred
                # to print new best params
                print('Best Params:', best_params)

        self.best_score = best_score
        self.best_params = best_params

        return best_ypred, best_idx


