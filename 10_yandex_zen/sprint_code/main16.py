#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go

note = '''
          Этот дашборд поможет вам выучить правила композиции дашбордов.
       '''

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'История игрового рынка'),

    html.Br(), # напишите код

    # пояснения
    html.Label(note),

    html.Br(), # напишите код
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)