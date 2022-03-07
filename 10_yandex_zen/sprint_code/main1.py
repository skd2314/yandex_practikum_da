#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd
import math

# задаём данные для отрисовки
x = range(-100, 100, 1)
x = [x / 10 for x in x]
y_sin = [math.sin(x) for x in x]
y_cos = [math.cos(x) for x in x]
data = [
        go.Scatter(x = pd.Series(x), y = pd.Series(y_sin), mode = 'lines', name = 'sin(x)'),
        go.Scatter(x = pd.Series(x), y = pd.Series(y_cos), mode = 'lines', name = 'cos(x)'), # напишите код
       ]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'Тригонометрические функции'),

dcc.Graph(
    figure = {
            'data': data,
            'layout': go.Layout(xaxis = {'title': 'x'},
                                yaxis = {'title': 'y'})
         },      
    id = 'trig_func'
),        
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)