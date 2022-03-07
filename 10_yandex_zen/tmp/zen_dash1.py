#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# импортируем необходимые библиотеки
 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
 
import plotly.graph_objs as go
 
from datetime import datetime
 
import pandas as pd
 
from sqlalchemy import create_engine
 
#подключение к базе данных для Postresql
db_config = {'user': 'my_user','pwd': 'my_user_password', 'host': 'localhost', 'port': 5432, 'db': 'zen'}
engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                            db_config['pwd'],
                                                            db_config['host'],
                                                            db_config['port'],
                                                            db_config['db']))
 
# получаем сырые данные и приводим столбец со временем к необходимому форм типу данных
query = '''
           SELECT * FROM dash_engagement
       '''
dash_engagement_df = pd.io.sql.read_sql(query, con = engine)
dash_engagement_df['dt'] = pd.to_datetime(dash_engagement_df['dt'])
 
query = '''
           SELECT * FROM dash_visits
       '''
dash_visits_df = pd.io.sql.read_sql(query, con = engine)
dash_visits_df['dt'] = pd.to_datetime(dash_visits_df['dt'])
 
# задаём лейаут
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, compress=False)
app.layout = html.Div(children=[  
   
    # формируем html
    html.H1(children = 'Анализ пользовательского взаимодействия с карточками статей на Яндекс.Дзен'),
    html.Br(),  
    html.Div('''Этот дашборд показывает статистику сервиса Яндекс.Дзен в разрезе: (а) количества взаимодействий
     пользователей с разбивкой по темам карточек, (б) количества карточек, которые генерируют источники с разными темами
     и (в) конверсия пользователей из просмотра карточек в просмотр статей. Используйте выбор временного периода, возрастные категории и темы карточек. '''),
    html.Br(),
 
    # Input; создаем элементы управления
 
    html.Div([  
 
        html.Div([
            # Создаем фильтр по времени
            
            html.Label('Временной период:'),
            dcc.DatePickerRange(
                start_date = dash_visits_df['dt'].min(),
                end_date = dash_visits_df['dt'].max(),
                display_format = 'YYYY-MM-DD',
                id = 'dt_selector',      
            ),
           
            # Создаем фильтр по возрастной категории
            html.Br(),
            html.Label('Возрастные категории:'),
            dcc.Dropdown(
                options = [{'label': x, 'value': x} for x in dash_visits_df['age_segment'].unique()],
                value = dash_visits_df['age_segment'].unique().tolist(),
                multi = True,
                id = 'selected_ages'
            ),
        ], className = 'six columns'),
    
        html.Div([ 
                                      
            # Создаем фильтр темы карточек
            html.Label('Темы карточек:'),
            dcc.Dropdown(
                options = [{'label': x, 'value': x} for x in dash_visits_df['item_topic'].unique()],
                value = dash_visits_df['item_topic'].unique().tolist(),
                multi = True,
                id = 'selected_item_topics'
            ),                
        ], className = 'six columns'),
  
    ], className = 'row'),
   
    html.Br(),
 
    # Output
 
    html.Div([
        html.Div([
            html.Br(),
          
            # Создаем график истории событий по темам карточек
            html.Label('Истории событий по темам карточек:'),    
       
            dcc.Graph(
                id = 'all_events_by_topics',
                style = {'height':'50vw'},
            ),  
        ], className = 'six columns'),            
 
        html.Div([
            html.Br(),
            # график c разбивкой по темам карточек
            html.Label('Распределение по темам:'),    
       
            dcc.Graph(
                id = 'broken_by_topic',
                style = {'height':'25vw'},
            ),  
 
            # график средней глубины взимодействия
            html.Label('Конверсия:'),    
       
            dcc.Graph(
                id = 'engagement',
                style = {'height':'25vw'},
            ),  
        ], className = 'six columns'),            
 
             
    ], className = 'row'),  
 
 
    ])
 
# Задаём логику дашборда
@app.callback(
    [Output('all_events_by_topics', 'figure'),
    Output('broken_by_topic', 'figure'),
    Output('engagement', 'figure'),
    ],
    [Input('selected_item_topics', 'value'),
    Input('selected_ages', 'value'),
    Input('dt_selector', 'start_date'),
    Input('dt_selector', 'end_date'),
    ])
 
 
def update_figures(selected_item_topics, selected_ages, start_date, end_date):
  # фильтруем данные
    filtered_visits = dash_visits_df.query('item_topic in @selected_item_topics')
    filtered_visits = filtered_visits.query('age_segment in @selected_ages')
    filtered_visits = filtered_visits.query('dt >= @start_date and dt <= @end_date')
 
 
    filtered_engagement = dash_engagement_df.query('item_topic in @selected_item_topics')
    filtered_engagement = filtered_engagement.query('age_segment in @selected_ages')
    filtered_engagement = filtered_engagement.query('dt >= @start_date and dt <= @end_date')
 
    # группируем данные
    history_grouped_topic_dt = (filtered_visits.groupby(['item_topic', 'dt'])
                                .agg({'visits': 'sum'})
                                .reset_index()
                                )
 
    grouped_by_source_topic = (filtered_visits.groupby(['source_topic'])
                                .agg({'visits': 'sum'})
                                .reset_index()
                            )
 
    engagement_by_event = (filtered_engagement.groupby(['event'])
                                .agg({'unique_users': 'mean'})
                                .sort_values(by = 'unique_users', ascending = False)
                                .rename(columns = {'unique_users': 'avg_unique_users'})
                                .reset_index()
                            )
 
    # Нормируем относительно среднего % от показов
    engagement_by_event['avg_unique_users'] = ((engagement_by_event['avg_unique_users'] / engagement_by_event['avg_unique_users'].max())*100).round(2)
    
    # Создааем графики
    # исторический график всех событий
 
    by_item_topic_plot = []
 
    for item_topic in history_grouped_topic_dt['item_topic'].unique():
        by_item_topic_plot += [go.Scatter(x = history_grouped_topic_dt.query('item_topic == @item_topic')['dt'],
                                    y = history_grouped_topic_dt.query('item_topic == @item_topic')['visits'],
                                    mode = 'lines',
                                    stackgroup = 'one',
                                    name = item_topic)]
 
    # график с разбивкой по темам источников
 
    by_source_topic_plot = [go.Pie(labels = grouped_by_source_topic['source_topic'],
                                    values = grouped_by_source_topic['visits'],
                                    name = 'source_topics')]
 
    # график глубины взаимодействия
 
    by_event_plot = [go.Bar(x = engagement_by_event['event'],
                            y = engagement_by_event['avg_unique_users'],
                            name = 'events')]
 
     # формируем результат для отображения
 
    return  (   # все исторические события по темам
                {
                'data': by_item_topic_plot,
                'layout': go.Layout(xaxis = {'title': 'Время'},
                                    yaxis = {'title': ''})
                },
                # разбивка по темам
                {
                'data': by_source_topic_plot,
                'layout': go.Layout()
                },            
                # глубина взаимодействия
                {
                'data': by_event_plot,
                'layout': go.Layout(xaxis = {'title': 'Действие'},
                                    yaxis = {'title': 'Количество'})
                },            
            )
 
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')