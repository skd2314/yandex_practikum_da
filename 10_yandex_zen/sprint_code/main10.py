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

# преобразуем типы
games_raw['year_of_release'] = pd.to_datetime(games_raw['year_of_release'])
columns = ['na_players', 'eu_players', 'jp_players', 'other_players']
for column in columns: games_raw[column] = pd.to_numeric(games_raw[column], errors = 'coerce')
games_raw['total'] = games_raw[['na_players',
                                'eu_players',
                                'jp_players',
                                'other_players']].sum(axis = 1).round(2)

# сформируйте графики для отрисовки
games_raw = games_raw[['name', 'platform', 'genre', 'total']].sort_values(by = 'total', ascending = False).head(10)

data = [go.Table(header = {'values': ['<b>Игра</b>', '<b>Платформа</b>',
                                      '<b>Жанр</b>', '<b>Продажи по всем регионам</b>'],
                           'fill_color': 'lightgrey',
                           'align': 'center'},
                 cells = {'values':games_raw.T.values})]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем заголовок тегом HTML
    html.H1(children = 'Топ-10 игр по продажам'),

    dcc.Graph(
        figure = {'data': data,
                  'layout': go.Layout()
                 },
        id = 'games_by_genre'
    ),         
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
