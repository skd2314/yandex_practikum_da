# !/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

# задаём данные для отрисовки
from sqlalchemy import create_engine

# пример подключения к базе данных для Postresql
# db_config = {'user': 'my_user',
#             'pwd': 'my_user_password',
#             'host': 'localhost',
#             'port': 5432,
#             'db': 'games'}
# engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
#                                                            db_config['pwd'],
#                                                            db_config['host'],
#                                                            db_config['port'],
#                                                            db_config['db']))
# пример подключения к базе данных для Sqlite
engine = create_engine('sqlite:////db/games.db', echo = False)

# получаем сырые данные
# напишите код
query = '''
            SELECT * FROM data_raw
        '''
games_raw = pd.io.sql.read_sql(query, con = engine)

# формируем данные для отчёта
games_grouped = (games_raw.groupby(['genre', 'year_of_release'])
                          .agg({'name':'count'})
                          .reset_index()
                          .rename(columns = {'name':'Games Released'})
                )

# формируем графики для отрисовки
data_games_by_year = []
for genre in games_grouped['genre'].unique():
    current = games_grouped.query('genre == @genre')
    data_games_by_year += [go.Scatter(x = current['year_of_release'],
                                    y = current['Games Released'],
                                    mode = 'lines',
                                    stackgroup = 'one',
                                    name = genre)] # напишите код



# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем html
    html.H1(children = 'Выпуск игр по годам'),

    dcc.Graph(
        figure = {'data': data_games_by_year,
                  'layout': go.Layout(xaxis = {'title': 'Год'},
                                      yaxis = {'title': 'Выпущенные игры'})
                 },
        id = 'games_by_year'
    ),         
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)