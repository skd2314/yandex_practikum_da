#!/usr/bin/python
# -*- coding: utf-8 -*-

# импортируем библиотеки
import pandas as pd
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

#print(dash_visits.head(5))


list_df = (dash_visits,)
for list_df in list_df:
    print('______________info_____________________')
    print(list_df.info())
    print('______________describe_________________')
    print(list_df.describe())
    print('______________head_____________________')
    print(list_df.head(10))
    print('______________sample___________________')
    print(list_df.sample(20))
    print('______________tail_____________________')
    print(list_df.tail(10))
    print('_____________isna______________________')
    print(list_df.isna().sum().reset_index())
    print('_____________duplicated________________')
    print(list_df.duplicated().sum())
    print('__количество пропущенных значений по каждому из столбцов__')
    print(list_df.isnull().mean().reset_index())