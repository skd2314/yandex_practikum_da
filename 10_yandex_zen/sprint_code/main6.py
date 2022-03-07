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
columns = ['user_score', 'critic_score']
for column in columns: games_raw[column] = pd.to_numeric(games_raw[column], errors = 'coerce')

# задаём цвета для рейтингов
games_raw['rating'] = games_raw['rating'].fillna('Неопр.')
rating_styles = {'E': {'color': 'red'},
                 'T': {'color': 'green'},
                 'M': {'color': 'blue'},
                 'E10+': {'color': 'magenta'},
                 'EC': {'color': 'yellow'},
                 'RP': {'color': 'orange'},
                 'AO': {'color': 'blue'},
                 'K-A': {'color': 'olive'},
                 'Неопр.': {'color': 'grey'}}
games_raw['rating_color'] = games_raw['rating'].apply(lambda x: rating_styles[x]['color'])

# задаём текст для отображения
games_raw['text'] = games_raw.apply(lambda x: '{}:{}'.format(x['platform'], x['name']), axis = 1)


# формируем графики для отрисовки
data = []
for rating in games_raw['rating'].unique():
    current = games_raw.query('rating == @rating')
    data += [go.Scatter(x = current['user_score'],
                        y = current['critic_score'],
                        mode = 'markers',
                        opacity = 0.5,
                        marker = {'color': rating_styles[rating]['color']},
                        text = games_raw['text'],
                        name = rating)]

# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
    
    # формируем заголовок тегом HTML
    html.H1(children = 'Игры по оценкам и возрастному рейтингу'),

    dcc.Graph(
        figure = {'data': data,
                  'layout': go.Layout(xaxis = {'title': 'Оценка игроков'},
                                      yaxis = {'title': 'Оценка критиков'},
                                      hovermode = 'closest'
                                      )
                 },
        id = 'score_scatter'
    ),         
 
])

# описываем логику дашборда
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
