import pandas as pd
import requests  # Импорт библиотеки для запросов к серверу
from bs4 import BeautifulSoup # Импорт библиотеки для автоматического парсинга странички

URL = 'https://code.s3.yandex.net/learning-materials/data-analyst/festival_news/index.html'
req = requests.get(URL)  # GET-запрос
soup = BeautifulSoup(req.text, 'lxml')

table = soup.find('table', attrs={'id': 'best_festivals'})

festivals_th  = []  # получим заголовоки из <th>
for row in table.find_all('th'):
    festivals_th.append(row.text)

festivals_tr = [] # получим название, место и дату из <td>
for row in table.find_all('tr'):
    if not row.find_all('th'):
        festivals_tr.append([element.text for element in row.find_all('td')])

#        festivals_tr.append(row.text)
    

# Датафрейм с данными
festivals = pd.DataFrame(festivals_tr, columns = festivals_th)
#festivals['festivals'] = festivals_tr
print(festivals)

