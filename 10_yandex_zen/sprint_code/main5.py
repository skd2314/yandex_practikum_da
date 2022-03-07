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
columns = ['na_players', 'eu_players', 'jp_players', 'other_players', 'user_score', 'critic_score']
for column in columns: games_raw[column] = pd.to_numeric(games_raw[column], errors = 'coerce')

# формируем данные для отчёта
games_grouped = (games_raw.groupby(['year_of_release'])
                          .agg({'na_players':'sum',  
                                'eu_players': 'sum',
                                'jp_players': 'sum',
                                'other_players': 'sum'})
                          .reset_index()
                )

# задаём настройки стилей для отрисовки в цикле
line_styles = {'na_players': {'color': 'red'},
               'eu_players': {'color': 'green'},
               'jp_players': {'color': 'blue'},
               'other_players': {'color': 'orange'}}

# формируем графики для отрисовки
data_games_by_year = []
for column in line_styles.keys():
    data_games_by_year += [go.Scatter(x = games_grouped['year_of_release'],
                                      y = games_grouped[column],
                                      mode = 'lines',
                                      line = line_styles[column],
                                      stackgroup = 'one',
                                      name = column)]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'Продажи игр по годам'),

    dcc.Graph(
        figure = {'data': data_games_by_year,
                  'layout': go.Layout(xaxis = {'title': 'Год'},
                                      yaxis = {'title': 'Продажи'})
                 },
        id = 'sales_by_year'
    ),         
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
