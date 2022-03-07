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
            SELECT * FROM data_raw
        '''
games_raw = pd.io.sql.read_sql(query, con = engine)
 
# преобразуем типы
games_raw['year_of_release'] = pd.to_datetime(games_raw['year_of_release'])
columns = ['na_players', 'eu_players', 'jp_players', 'other_players']
for column in columns: games_raw[column] = pd.to_numeric(games_raw[column], errors = 'coerce')
 
# формируем данные для отчёта
games_grouped = (games_raw.groupby(['year_of_release', 'genre'])
                          .agg({'name': 'nunique'})
                          .reset_index()
                          .rename(columns = {'name': 'games_launched'})
                )
 
# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,compress=False)
app.layout = html.Div(children=[  
 
    # формируем заголовок тегом HTML
    html.H1(children = 'Выпуск игр по годам'),
 
    # выбор временного периода
    html.Label('Временной период:'),
    dcc.DatePickerRange(
        start_date = games_grouped['year_of_release'].dt.date.min(),
        end_date = datetime(2016,1,1).strftime('%Y-%m-%d'),
        display_format = 'YYYY-MM-DD',
        id = 'dt_selector',       
    ),  
 
    # выбор режима отображения абсолютные/относительные значения
    html.Label('Режим отображения:'),
    dcc.RadioItems( # напишите код
        options = [
            {'label': 'Абсолютные значения', 'value': 'absolute_values'},
            {'label': '% от общего выпуска', 'value': 'relative_values'},
        ],
        value = 'absolute_values',
        id = 'mode_selector' # напишите код
    ),       
 
    # график выпуска игр по годам
    dcc.Graph(
        id = 'sales_by_year'
    ),         
 
])
 
# описываем логику дашборда
@app.callback(
    [Output('sales_by_year', 'figure'),
    ],
    [Input('dt_selector', 'start_date'),
     Input('dt_selector', 'end_date'),
     Input('mode_selector', 'value'),
    ])
def update_figures(start_date, end_date, mode):
 
    # приводим входные параметры к нужным типам
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
 
    # применяем фильтрацию
    filtered_data = games_grouped.query('year_of_release >= @start_date and year_of_release <= @end_date')
 
    # трансформируем в соотв. с выбранным режимом отображения
    if mode == 'relative_values':
        total_by_year = (filtered_data.groupby('year_of_release')
                         .agg({'games_launched': 'sum'})
                         .rename(columns = {'games_launched': 'total'})
                        )
        filtered_data = (filtered_data.set_index('year_of_release')
                         .join(total_by_year)
                         .reset_index())
        filtered_data['games_launched'] = filtered_data['games_launched'] / filtered_data['total']
 
        # формируем графики для отрисовки с учётом фильтров
    data = []
    for genre in filtered_data['genre'].unique():
        data += [go.Scatter(x = filtered_data.query('genre == @genre')['year_of_release'],
                            y = filtered_data.query('genre == @genre')['games_launched'],
                            mode = 'lines',
                            stackgroup = 'one',
                            name = genre)]
 
  # формируем результат для отображения
    return (
        {
            'data': data,
            'layout': go.Layout(xaxis = {'title': 'Дата и время'},
                                yaxis = {'title': 'Выпущенные игры'})
        },
    )  
 
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
    
    
    