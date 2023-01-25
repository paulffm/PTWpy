import os
import math
import pandas as pd
import numpy as np
#import orjson

feedVar = '/NC/_N_CH_GD9_ACX/SYG_S9|3'
measurementVar = '/NC/_N_CH_GD9_ACX/SYG_S9|4'
axisVar = '/NC/_N_CH_GD9_ACX/SYG_I9|2'
loopVar1 = '/NC/_N_CH_GD9_ACX/SYG_I9|3'
loopVar2 = '/NC/_N_CH_GD9_ACX/SYG_I9|4'
loopVar3 = '/NC/_N_CH_GD9_ACX/SYG_I9|5'
msgVar = '/Channel/ProgramInfo/msg|u1'
blockVar = "/Channel/ProgramInfo/block|u1.2"

date = "2022-07-28"
file = r'/Users/paulheller/PycharmProjects/PTWKohn/Data/2022-07-28_MRM_DMC850_20220509.MPF.csv'

data = pd.read_csv(file, parse_dates=[0], low_memory=False)

data['DateTime'] = pd.to_datetime(data["_time"])
data['Date'] = data['DateTime'].dt.strftime('%Y-%m-%d')

data = data.sort_values(by="CYCLE")

active_axis=1
machine = "DMC850"
measurement="Lineare_Referenzfahrt_Feed"

dataFilter = data.sort_values(by="CYCLE")
dataFilter = dataFilter.fillna(method="ffill")
dataFilter = dataFilter.loc[dataFilter[axisVar]==active_axis]
dataFilter = dataFilter.loc[dataFilter[measurementVar].str.contains(measurement)]

dataFilter = dataFilter[0:10000]

dataFilter["CYCLE"].diff().describe()
dataFilter.columns

input_Array = [f'DES_POS|{active_axis}', f'VEL_FFW|{active_axis}']



# %% Visualisierungen

import plotly.express as px
import plotly.graph_objects as go

#### Visualisierungen aller Daten: Vergleich Ist- und Vorhersagewerte der Trainings- und Testdaten

# .values all values listed as array
# .diff change between periods
# .shift index shift of
# .fillna(0) set NaN values to 0

fig = go.Figure()

fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=dataFilter[f"TORQUE_FFW|{active_axis}"],
                         name=f'TORQUE_FFW|{active_axis}', mode='markers'))
fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=dataFilter[f"VEL_FFW|{active_axis}"].values,
                         name=f'VEL_FFW|{active_axis}', mode='markers'))

fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=dataFilter[f"DES_POS|{active_axis}"],
                         name=f'DES_POS|{active_axis}', mode='markers'))
fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=(dataFilter[f"DES_POS|{active_axis}"].diff()/0.002).shift(periods=10).fillna(0.0).values,
                         name=f'n_DES_POS|{active_axis}', mode='markers'))
fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=(dataFilter[f"VEL_FFW|{active_axis}"].diff()/0.002*0.00147/0.030*1000*16.33).fillna(0.0).values,
                         name=f'VEL_FFW|{active_axis}_diff', mode='markers'))
fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=(dataFilter[f"DES_POS|{active_axis}"].diff().diff()/0.002/0.002*0.00147/0.030*1000*12).fillna(0.0).values,
                         name=f'n_DES_POS|{active_axis}', mode='markers'))
fig.add_trace(go.Scattergl(x=dataFilter["CYCLE"].values, y=((dataFilter[f"ENC2_POS|{active_axis}"].diff()/0.002).diff()/0.002*0.00147/0.030*1000*12).fillna(0.0).values,
                         name=f'n_DES_POS|{active_axis}', mode='markers'))

#fig.write_html(os.path.join(r'C:\Users\O.Kohn_Lokal\Desktop','fig.html'))
fig.show()