#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

# задаём данные для отрисовки
from sqlalchemy import create_engine

# пример подключения к базе данных для Postresql
#db_config = {'user': 'my_user',
#             'pwd': 'my_user_password',
#             'host': 'localhost',
#             'port': 5432,
#             'db': 'games'}
#engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
#                                                            db_config['pwd'],
#                                                            db_config['host'],
#                                                            db_config['port'],
#                                                            db_config['db']))
# пример подключения к базе данных для Sqlite
engine = create_engine('sqlite:////db/games.db', echo = False)

# получаем сырые данные
query = '''
            SELECT * FROM data_raw
        '''
games_raw = pd.io.sql.read_sql(query, con = engine)

# формируем данные для отчёта
games_grouped = (games_raw.groupby(['genre'])
                          .agg({'name': 'nunique'})
                          .rename(columns = {'name': 'games_launched'})
                          .reset_index()
                )
games_grouped = games_grouped.sort_values(by='games_launched', ascending=False) # напишите код

# формируем подписи
games_grouped['label'] = games_grouped.apply(lambda x: '{} игр'.format(x['games_launched']), axis = 1)

# формируем графики для отрисовки
data = [go.Bar(x = games_grouped['genre'],
               y = games_grouped['games_launched'],
               text = games_grouped['label'], # напишите код
               textposition = 'auto',               
               name = 'games_launched')]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем заголовок тегом HTML
    html.H1(children = 'Выпуск игр по жанрам'),

    dcc.Graph(
        figure = {'data': data,
                  'layout': go.Layout(xaxis = {'title': 'Жанр'},
                                      yaxis = {'title': 'Выпущенные игры'})
                 },
        id = 'launches_by_genre' # напишите код
    ),         
 
])
 
# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
