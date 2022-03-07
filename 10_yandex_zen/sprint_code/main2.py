#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

# задаём данные для отрисовки
games_raw = pd.read_csv('/datasets/games_full.csv')
games_raw['Year_of_Release'] = pd.to_datetime(games_raw['Year_of_Release'])

# формируем данные для отчёта
games_grouped = (games_raw.groupby(['Genre', 'Year_of_Release']) # ваш код
                          .agg({'Name':'count'}) # ваш код
                          .reset_index()
                          .rename(columns = {"Name":'Games Released'}) # ваш код
                )

# формируем графики для отрисовки
data = []
for genre in games_grouped['Genre'].unique():
    current = games_grouped.query('Genre == @genre')
    data += [go.Scatter(x = current["Year_of_Release"], # напишите код
                        y = current["Games Released"], # напишите код
                        mode = 'lines',
                        stackgroup = 'one', # напишите код
                        name = genre)]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'Выпуск игр по годам'),

    dcc.Graph(
        figure = {'data': data,
                  'layout': go.Layout(xaxis = {'title': 'Год'},
                                      yaxis = {'title': 'Выпущенные игры'})
                 },
        id = 'games_by_year' # ваш код
    ),      
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)