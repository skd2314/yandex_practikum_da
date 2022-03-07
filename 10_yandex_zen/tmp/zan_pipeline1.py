#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# Импортируем необходимые библиотеки: 
import pandas as pd
from sqlalchemy import create_engine
import sys
import math
import getopt
from datetime import datetime
 
 
if __name__ == "__main__":  

#Задаём входные параметры
    unixOptions = "sdt:edt"  
    gnuOptions = ["start_dt=", "end_dt="]
 
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]    
    try:  
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:  
   
        print (str(err))
        sys.exit(2)
   
    start_dt = '2019-09-24 18:00:00'
    end_dt = '2019-09-24 19:00:00'  
    for currentArgument, currentValue in arguments:  
        if currentArgument in ("-sdt", "--start_dt"):
            start_dt = currentValue                                  
        elif currentArgument in ("-edt", "--end_dt"):
            end_dt = currentValue        
 
 
 # Задаём параметры подключения к БД
    db_config = {'user': 'my_user',
                  'pwd': 'my_user_password',
                 'host': 'localhost',
                 'port': 5432,
                   'db': 'zen'}  
                   
 # Формируем строку соединения с БД и подключаемся к ней:
    connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                             db_config['pwd'],
                                                             db_config['host'],
                                                             db_config['port'],
                                                             db_config['db'])                
 
    engine = create_engine(connection_string)    
 
# Формируем sql-запрос:
 
    # Теперь выберем из таблицы только те строки,
    # которые были выпущены между start_dt и end_dt 
 
    query = ''' SELECT event_id, age_segment, event, item_id, item_topic, item_type, source_id, source_topic, source_type, date_trunc('minute', TO_TIMESTAMP(ts / 1000) AT TIME ZONE 'Etc/UTC') as dt, user_id
            FROM log_raw
            WHERE TO_TIMESTAMP(ts / 1000) AT TIME ZONE 'Etc/UTC' BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
            '''.format(start_dt, end_dt)
   
 
 
    log_raw = pd.io.sql.read_sql(query, con = engine)
    print('В таблице log_raw',log_raw.shape[0],'записей.')
    print(log_raw.info())
 
 
 # Преобразуем данные к нужным типам
    columns_str=['age_segment', 'event', 'item_topic', 'item_type', 'source_topic', 'source_type']
    columns_numeric = ['event_id', 'item_id', 'source_id', 'user_id']
    columns_datetime = ['dt']
 
    for column in columns_str: log_raw[column] = log_raw[column].astype(str)
    for column in columns_numeric: log_raw[column] = pd.to_numeric(log_raw[column], errors='coerce')
    for column in columns_datetime: log_raw[column] = pd.to_datetime(log_raw[column]).dt.round('min')
 
 
 #Готовим агрегирующие таблицы:
    dash_visits = log_raw.groupby(['item_topic', 'source_topic', 'age_segment', 'dt']).agg({'event':'count'}).reset_index()
    dash_engagement = log_raw.groupby(['item_topic','age_segment','dt','event']).agg({'user_id':'nunique'}).reset_index()

 
    dash_visits = dash_visits.rename(columns = {'event': 'visits'})
    dash_engagement = dash_engagement.rename(columns = {'user_id': 'unique_users'})
    
    print(dash_visits.head(5))
    print('В таблице dash_visits',dash_visits.shape[0],'записей.')
    print(dash_visits.info())
    print(dash_engagement.head(5))
    print('В таблице dash_engagement',dash_engagement.shape[0],'записей.')
    print(dash_engagement.info())
    
   
    tables = {'dash_visits': dash_visits,
              'dash_engagement': dash_engagement}
 
    for table_name, table_data in tables.items():
        query = '''DELETE FROM {}
                   WHERE dt BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
                '''.format(table_name,start_dt, end_dt)
        engine.execute(query)
        table_data.to_sql(name = table_name, con = engine, if_exists = 'append', index = False)
        print('таблица', table_name, 'обновлена')