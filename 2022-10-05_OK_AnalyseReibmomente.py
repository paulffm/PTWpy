import math
import pandas as pd
import numpy as np
import pickle

feedVar = '/NC/_N_CH_GD9_ACX/SYG_S9|3'
measurementVar = '/NC/_N_CH_GD9_ACX/SYG_S9|4'
axisVar = '/NC/_N_CH_GD9_ACX/SYG_I9|2'
loopVar1 = '/NC/_N_CH_GD9_ACX/SYG_I9|3'
loopVar2 = '/NC/_N_CH_GD9_ACX/SYG_I9|4'
loopVar3 = '/NC/_N_CH_GD9_ACX/SYG_I9|5'
msgVar = '/Channel/ProgramInfo/msg|u1'
blockVar = "/Channel/ProgramInfo/block|u1.2"

date = "2022-07-28"
file = r'C:\Users\O.Kohn_Lokal\Desktop\ConditionMonitoring\DMC850\2022-07-28_MRM_\2022-07-28_MRM_DMC850_20220509.MPF.csv'

data = pd.read_csv(file, parse_dates=[0])

data['DateTime'] = pd.to_datetime(data["_time"])
data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')

data = data.sort_values(by="CYCLE")

active_axis=1
machine = "DMC850"
measurement="Lineare_Referenzfahrt_Feed"

dataFilter = data.sort_values(by="CYCLE")
dataFilter = dataFilter.fillna(method="ffill")
'''dataFilter = dataFilter.loc[dataFilter[axisVar]==active_axis]
dataFilter = dataFilter.loc[dataFilter[measurementVar].str.contains(measurement)]

dataFilter[loopVar2].unique()
dataFilter[f'DES_POS_-1|{active_axis}'] = dataFilter[f'DES_POS|{active_axis}'].shift(-1)
dataFilter[f'DES_POS_-2|{active_axis}'] = dataFilter[f'DES_POS|{active_axis}'].shift(-2)
dataFilter[f'DES_POS_-3|{active_axis}'] = dataFilter[f'DES_POS|{active_axis}'].shift(-3)
dataFilter[f'DES_POS_-4|{active_axis}'] = dataFilter[f'DES_POS|{active_axis}'].shift(-4)
dataFilter[f'DES_POS_-5|{active_axis}'] = dataFilter[f'DES_POS|{active_axis}'].shift(-5)

dataFilter = dataFilter[5:-5]'''

input_Array = [f'VEL_FFW|{active_axis}', f'TORQUE_FFW|{active_axis}']

input = dataFilter[input_Array].copy()
output = dataFilter[[f'CURRENT|{active_axis}']].copy()

## @ToDo:

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scalerInput = MinMaxScaler()
scalerInput = scalerInput.fit(input)
scalerOutput = MinMaxScaler()
scalerOutput = scalerOutput.fit(output)
pickle.dump(scalerInput, open(r"C:\Users\O.Kohn_Lokal\Desktop\scalerInput.sav", 'wb'))
pickle.dump(scalerOutput, open(r"C:\Users\O.Kohn_Lokal\Desktop\scalerOutput.sav", 'wb'))



X_train, X_test, y_train, y_test = train_test_split(scalerInput.transform(input),
                                                    scalerOutput.transform(output),
                                                    train_size=0.8)
print("Training")
#regr = MLPRegressor(max_iter=200000, hidden_layer_sizes=(200,300,200)).fit(X_train, y_train.ravel())
regr = MLPRegressor(max_iter=200000).fit(X_train, y_train.ravel())
print("Trained")
pickle.dump(regr, open(r"C:\Users\O.Kohn_Lokal\Desktop\regr.sav", 'wb'))


# %% Visualisierungen

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#### Visualisierungen aller Daten: Vergleich Ist- und Vorhersagewerte der Trainings- und Testdaten
fig = go.Figure()
'''fig.add_trace(go.Scattergl(x=y_train[:,0], y=scalerOutput.inverse_transform([regr.predict(X_train)]).ravel(),
                         name=f'CURRENT|{active_axis}_train', mode='markers'))'''
fig.add_trace(go.Scattergl(x=scalerOutput.inverse_transform(y_test).ravel(), y=scalerOutput.inverse_transform([regr.predict(X_test)]).ravel(),
                         name=f'CURRENT|{active_axis}_test', mode='markers'))
fig.show()


#### Visualisierungen aller Daten:

prediction = scalerOutput.inverse_transform([regr.predict(scalerInput.transform(dataFilter[input_Array]))]).ravel()

fig = go.Figure()
'''fig.add_trace(go.Scatter(x=dataFilter[f"ENC2_POS|{active_axis}"], y=dataFilter[f"CURRENT|{active_axis}"], name=f'CURRENT|{active_axis}',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=dataFilter[f"ENC2_POS|{active_axis}"], y=prediction, name = f'CURRENT|{active_axis} predicted',
                         line=dict(color='royalblue', width=4)))'''
fig.add_trace(go.Scattergl(x=dataFilter.index, y=dataFilter[f"CURRENT|{active_axis}"], name=f'CURRENT|{active_axis}',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scattergl(x=dataFilter.index, y=prediction, name = f'CURRENT|{active_axis} predicted',
                         line=dict(color='royalblue', width=4)))
fig.show()


fig = go.Figure()
fig.add_trace(go.Scattergl(x=dataFilter[f"CURRENT|{active_axis}"], y=prediction,
                         name=f'CURRENT|{active_axis}', mode='markers'))
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=dataFilter[f"VEL_FFW|{active_axis}"], y=prediction,
                         name=f'CURRENT|{active_axis}', mode='markers'))
fig.show()