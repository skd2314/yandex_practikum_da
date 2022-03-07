#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from warnings import simplefilter

from datetime import datetime
import pandas as pd

# задаём данные для отрисовки
from sqlalchemy import create_engine

# Задаём параметры подключения к БД,
# их можно узнать у администратора БД.
db_config = {'user': 'praktikum_student', # имя пользователя
            'pwd': 'Sdf4$2;d-d30pp', # пароль
            'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
            'port': 6432, # порт подключения
            'db': 'data-analyst-zen-project-db'} # название базы данных

# Формируем строку соединения с БД.
connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                db_config['pwd'],
                                                db_config['host'],
                                                db_config['port'],
                                                db_config['db'])

# Подключаемся к БД.
engine = create_engine(connection_string) 

# Формируем sql-запрос.
# получаем сырые данные
query = '''
            SELECT * FROM dash_visits
        '''
dash_visits = pd.io.sql.read_sql(query, con = engine)



# Выполняем запрос и сохраняем результат
# выполнения в DataFrame.
# Sqlalchemy автоматически установит названия колонок
# такими же, как у таблицы в БД. Нам останется только
# указать индексную колонку с помощью index_col.
dash_visits.to_csv('dash_visits.csv', sep='\t', encoding='utf-8', index=False)


#table1 источники
zen_topic = dash_visits.groupby('source_topic').agg({'visits':'sum'})\
                                               .reset_index().sort_values(by='visits', ascending=False)
#zen_topic.head(20)

#table2 статьи
zen_item = dash_visits.groupby('item_topic').agg({'visits':'sum'})\
                                               .reset_index().sort_values(by='visits', ascending=False)
#zen_item.head(20)



##########################################################################################################################
#graf1
# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        # формируем заголовок тегом HTML
        html.H1(children='Анализ взаимодействия пользователей с карточками Яндекс.Дзен'),

        # график 1
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=zen_topic['source_topic'],
                        y=zen_topic['visits'],
                        name='zen_topic',
                        marker_color='#22a6b3'
                    )
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Источники'},
                    yaxis={'title': 'Количество'},
                    title={
                    'text': 'Темы Zen источников',
                    'y':0.85,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                ),
                },
            id='zen_topic_graf',
        ),
        
        # график 2
        dcc.Graph(
            figure={
                'data': [
                    go.Bar(
                        x=zen_item['item_topic'],
                        y=zen_item['visits'],
                        name='zen_item',
                        marker_color='#D06224'
                    )
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Статьи'},
                    yaxis={'title': 'Количество'},
                    title={
                    'text': 'Статьи Zen источников',
                    'y':0.85,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                ),
            },
            id='zen_item_graf',
        ),
        
        

        
        
    ]
)

#serv
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')












    