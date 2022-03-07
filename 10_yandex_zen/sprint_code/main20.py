#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go

from datetime import datetime

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
            SELECT * FROM agg_games_year_genre_platform
        '''
agg_games_year_genre_platform = pd.io.sql.read_sql(query, con = engine)
agg_games_year_genre_platform['year_of_release'] = pd.to_datetime(agg_games_year_genre_platform['year_of_release'])

query = '''
            SELECT * FROM agg_games_year_score
        '''
agg_games_year_score = pd.io.sql.read_sql(query, con = engine)
agg_games_year_score['year_of_release'] = pd.to_datetime(agg_games_year_score['year_of_release'])
# игнорируем записи без оценок
agg_games_year_score = agg_games_year_score.query('avg_user_score > 0 and avg_critic_score > 0')

note = '''
          Этот дашборд показывает историю игрового рынка (исключая мобильные устройства).
          Используйте выбор интервала даты выпуска, жанра и платформы для управления дашбордом.
          Используйте селектор выбора режима отображения для того, чтобы показать абсолютные
          или относительные значения выпуска и продаж игр по годам.
       '''

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'История игрового рынка'),

    # пояснения
    html.Label(note),
 
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)