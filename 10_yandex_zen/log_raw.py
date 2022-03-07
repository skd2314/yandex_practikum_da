#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

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

############
# Формируем sql-запрос.
# получаем сырые данные log_raw
query = '''
            SELECT * 
            FROM log_raw
        '''
log_raw = pd.io.sql.read_sql(query, con=engine, index_col='event_id')


# Выполняем запрос и сохраняем результат
# выполнения в DataFrame.
# Sqlalchemy автоматически установит названия колонок
# такими же, как у таблицы в БД. Нам останется только
# указать индексную колонку с помощью index_col.
log_raw.to_csv('log_raw.csv', index=False)



log_raw['event_time'] = pd.to_datetime(log_raw['ts'], unit = 'ms') 


##############################################################












    