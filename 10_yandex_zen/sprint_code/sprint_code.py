#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import getopt

import pandas as pd

if __name__ == "__main__":

    # Задаём определения входных параметров
    unixOptions = "r:" # напишите код
    gnuOptions = ["regions="] # напишите код

    # Читаем входные параметры
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]
    try:  
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:  
        print (str(err))
        sys.exit(2)

    # Обрабатываем входные параметры
    regions = 'Germany,France,Russia'.split(',')
    for currentArgument, currentValue in arguments:  
        if currentArgument in (gnuOptions): # ваш код здесь
            regions = currentValue.split(',') # ваш код здесь

    urbanization = pd.read_csv('/datasets/urbanization.csv')

    # Фильтруем и определяем максимальный уровень урбанизации
    urbanization = urbanization.query('Entity.isin(@regions)')
    urbanization = urbanization.groupby('Entity').agg({'Urban': 'max'}) # ваш код

    print(urbanization)
    
########################################################################################################
#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import getopt
from datetime import datetime

import pandas as pd

if __name__ == "__main__":

    # Задаём определения входных параметров
    unixOptions = "r:s:e:" # напишите код
    gnuOptions = ["regions=", "start_dt=", "end_dt="] # напишите код

    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]
    try:  
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:  
        print (str(err))
        sys.exit(2)

    # Обрабатываем входные параметры
    regions = 'Germany,France,Russia'.split(',')
    start_dt = '1998-01-01 00:00:00'
    end_dt = '1999-12-31 23:59:59'     
    for currentArgument, currentValue in arguments:  
        if currentArgument in ("-r", "--regions"):
            regions = currentValue.split(',')
        elif currentArgument in ("-s", "--start_dt"):
            start_dt = datetime.strptime(currentValue, '%Y-%m-%d %H:%M:%S') # напишите код
        elif currentArgument in ("-e", "--end_dt"):
            end_dt = datetime.strptime(currentValue, '%Y-%m-%d %H:%M:%S') # напишите код

    urbanization = pd.read_csv('/datasets/urbanization.csv')[['Entity', 'Year', 'Urban']]

    # Приводим колонки urbanization к нужным типам
    urbanization['Year'] = pd.to_datetime(urbanization['Year'], format = '%Y-%m-%d') # ваш код

    # Фильтруем и определяем максимальный уровень урбанизации
    urbanization = urbanization.query('Entity.isin(@regions) and Year >= @start_dt and Year <= @end_dt') # напишите код

    print(urbanization.sort_values(by = ['Entity', 'Year'], ascending=True)) # ваш код
    #################################################################################################################################
#!/usr/bin/python
import pandas as pd
from sqlalchemy import create_engine

db_config = {'user': 'my_user',
             'pwd': 'my_user_password',
             'host': 'localhost',
             'port': 5432,
             'db': 'games'}   
  
connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
    db_config['user'],
    db_config['pwd'],
    db_config['host'],
    db_config['port'],
    db_config['db'],
)

engine = create_engine(connection_string) 
        
query = '''
            select * from data_raw
        '''
data_raw = pd.io.sql.read_sql(query, con=engine, index_col='game_id')

print(data_raw.info())
print(data_raw.head(5))
#####################################################################################
#!/usr/bin/python
import sys

import getopt
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine

if __name__ == '__main__':

    # укажите входные параметры в строках
    unixOptions = 'sdt:edt:'
    gnuOptions = ['start_dt=', 'end_dt=']

    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]

    try:
        arguments, values = getopt.getopt(
            argumentList, unixOptions, gnuOptions
        )
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    start_dt = '1981-01-01'
    end_dt = '1998-01-01'
    for currentArgument, currentValue in arguments:
        if currentArgument in ('-sdt', '--start_dt'):
            start_dt = currentValue
        elif currentArgument in ('-edt', '--end_dt'):
            end_dt = currentValue

    db_config = {
        'user': 'my_user',
        'pwd': 'my_user_password',
        'host': 'localhost',
        'port': 5432,
        'db': 'games',
    }
    connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
        db_config['user'],
        db_config['pwd'],
        db_config['host'],
        db_config['port'],
        db_config['db'],
    )

    engine = create_engine(connection_string)

    query = ''' SELECT *
                FROM data_raw
                WHERE year_of_release::TIMESTAMP BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
            '''.format(start_dt, end_dt)
    
    engine.execute(query)
    
    data_raw = pd.io.sql.read_sql(query, con=engine, index_col='game_id')
    
    print(data_raw['year_of_release'].unique())
########################################################################################################
#!/usr/bin/python
import sys

import getopt
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine

if __name__ == "__main__":

    unixOptions = "sdt:edt:"  
    gnuOptions = ["start_dt=", "end_dt="]

    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:] 

    try:  
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:  
        print (str(err))
        sys.exit(2)

    start_dt = '1981-01-01'
    end_dt = '1998-01-01'   
    for currentArgument, currentValue in arguments:  
        if currentArgument in ("-sdt", "--start_dt"):
            start_dt = currentValue                                   
        elif currentArgument in ("-edt", "--end_dt"):
            end_dt = currentValue  

    db_config = {'user': 'my_user',
                 'pwd': 'my_user_password',
                 'host': 'localhost',
                 'port': 5432,
                 'db': 'games'}   
    connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                             db_config['pwd'],
                                                             db_config['host'],
                                                             db_config['port'],
                                                             db_config['db'])

    engine = create_engine(connection_string)
            
    query = ''' SELECT *
                FROM data_raw
                WHERE year_of_release::TIMESTAMP BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
            '''.format(start_dt, end_dt)

    data_raw = pd.io.sql.read_sql(query, con = engine, index_col = 'game_id')

    columns_str = ['name', 'platform', 'genre', 'rating']
    columns_numeric = ['na_players', 'eu_players', 'jp_players', 'other_players', 'critic_score', 'user_score']
    columns_datetime = ['year_of_release']

    for column in columns_str: data_raw[column] = data_raw[column].astype(str)  
    for column in columns_numeric: data_raw[column] = pd.to_numeric(data_raw[column], errors='coerce')
    for column in columns_datetime: data_raw[column] = pd.to_datetime(data_raw[column])   

    data_raw['total_copies_sold'] = data_raw[['na_players',
                          'eu_players',
                          'jp_players',
                          'other_players']].sum(axis = 1)

    agg_games_year_genre_platform = data_raw.groupby(['year_of_release', 'genre', 'platform']).agg({'name': 'count', 'total_copies_sold': 'sum'})
    agg_games_year_score = data_raw.groupby(['year_of_release', 'genre', 'platform']).agg({'critic_score': 'mean', 'user_score': 'mean'})

    print(data_raw.info())
    print(agg_games_year_genre_platform.head(5))
    print(agg_games_year_score.head(5))

    print(data_raw.info())
########################################################################################################################################################################
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

import getopt
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine

if __name__ == '__main__':

    # задаём входные параметры
    unixOptions = 'sdt:edt:'
    gnuOptions = ['start_dt=', 'end_dt=']

    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]  # excluding script name

    try:
        arguments, values = getopt.getopt(
            argumentList, unixOptions, gnuOptions
        )
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    start_dt = '1981-01-01'
    end_dt = '1998-01-01'
    for currentArgument, currentValue in arguments:
        if currentArgument in ('-sdt', '--start_dt'):
            start_dt = currentValue
        elif currentArgument in ('-edt', '--end_dt'):
            end_dt = currentValue

    db_config = {
        'user': 'my_user',
        'pwd': 'my_user_password',
        'host': 'localhost',
        'port': 5432,
        'db': 'games',
    }
    connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
        db_config['user'],
        db_config['pwd'],
        db_config['host'],
        db_config['port'],
        db_config['db'],
    )

    engine = create_engine(connection_string)

    query = ''' SELECT * 
                FROM data_raw 
                WHERE year_of_release::TIMESTAMP BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
            '''.format(
        start_dt, end_dt
    )

    data_raw = pd.io.sql.read_sql(query, con=engine, index_col='game_id')

    columns_str = ['name', 'platform', 'genre', 'rating']
    columns_numeric = [
        'na_players',
        'eu_players',
        'jp_players',
        'other_players',
        'critic_score',
        'user_score',
    ]
    columns_datetime = ['year_of_release']

    for column in columns_str:
        data_raw[column] = data_raw[column].astype(str)
    for column in columns_numeric:
        data_raw[column] = pd.to_numeric(data_raw[column], errors='coerce')
    for column in columns_datetime:
        data_raw[column] = pd.to_datetime(data_raw[column])
    data_raw['total_copies_sold'] = data_raw[
        ['na_players', 'eu_players', 'jp_players', 'other_players']
    ].sum(axis=1)

    agg_games_year_genre_platform = data_raw.groupby(
        ['year_of_release', 'genre', 'platform']
    ).agg({'name': 'count', 'total_copies_sold': 'sum'})
    agg_games_year_score = data_raw.groupby(
        ['year_of_release', 'genre', 'platform']
    ).agg({'critic_score': 'mean', 'user_score': 'mean'})

    agg_games_year_genre_platform = agg_games_year_genre_platform.rename(
        columns={'name': 'games'}
    )  # напишите код
    agg_games_year_score = agg_games_year_score.rename(
        columns={'critic_score': 'avg_critic_score', 'user_score': 'avg_user_score'}
    )  # напишите код

    agg_games_year_genre_platform = agg_games_year_genre_platform.fillna(
        0
    ).reset_index()
    agg_games_year_score = agg_games_year_score.fillna(0).reset_index()

    tables = {
        'agg_games_year_genre_platform': agg_games_year_genre_platform,  # ваш код
        'agg_games_year_score': agg_games_year_score,
    }  # ваш код

    for table_name, table_data in tables.items():

        query = '''
                  DELETE FROM {} WHERE year_of_release BETWEEN '{}'::TIMESTAMP AND '{}'::TIMESTAMP
                '''.format(
            table_name, start_dt, end_dt
        )  # напишите код
        engine.execute(query)

        table_data.to_sql(
            name=table_name, con=engine, if_exists='append', index=False
        )  # напишите код

    print('All done.')
############################################################################################################

    
    
    
    
    