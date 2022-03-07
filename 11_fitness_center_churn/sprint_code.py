import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# прочитайте данные с атрибутами аккаунтов компаний и активностью
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep=';')

print(fb.shape)
print(fb.head())
################################################################################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# прочитайте данные с атрибутами аккаунтов компаний и активностью
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop(['Total Interactions'], axis = 1)
y = fb['Total Interactions']
################################################################################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# прочитайте данные с атрибутами аккаунтов компаний и активностью
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# задаём алгоритм для модели 
model = RandomForestRegressor()

# обучите модель
# напишите свой код здесь
model.fit(X, y)

# сделайте прогноз обученной моделью
predictions = model.predict(X)

# нарисуем график прогноз-факт
sns.scatterplot(y, predictions, s = 15, alpha = 0.6)
plt.title('График Прогноз-Факт')
plt.ylabel('Прогноз')
plt.xlabel('Факт')
plt.show()
################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split

# прочитайте данные с атрибутами аккаунтов компаний на facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# прочитайте данные с атрибутами аккаунтов компаний на facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# зададим алгоритм для нашей модели 
model = RandomForestRegressor(random_state=0) 

# обучим модель
model.fit(X_train, y_train) # обучите вашу модель на обучающей выборке

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test)

# оценим метрику R-квадрат на валидационной выборке и напечатаем
r2 = r2_score(y_test, predictions)
print('Значение метрики R-квадрат: ', r2)
##################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# прочитайте данные с атрибутами аккаунтов компаний на facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделите данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# гистограмма целевой переменной на train
sns.distplot(y_train) 

# гистограмма целевой переменной на test
sns.distplot(y_test) 
###################################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# прочитайте данные с атрибутами аккаунтов компаний на facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# корреляционная матрица
corr_m = fb.corr()

sns.heatmap(corr_m, square = True, annot = True)
plt.figure(figsize = (15, 15))
##################################################################################################

import numpy as np

# Создайте векторы x и y
x = [12, 9, 4, 5, 5, 7, 7, 6, 7, 11, 9, 8, 10, 4, 11]
y = [28.1, 18.7, 1.0, 10.2, 11.6, 19.9, 24.4, 18.1, 18.5, 25.0, 21.8, 13.4, 18.0, 11.1, 21.1]

y_1 = [(2 + 2*i) for i in x] # рассчитайте прогноз первой функцией
y_2 = [(3 + 1*i) for i in x] # рассчитайте прогноз второй функцией

print('Прогноз первой моделью:', y_1)
print('Прогноз второй моделью:', y_2)
##################################################################################################
import numpy as np

# Создайте векторы x и y
x = [12, 9, 4, 5, 5, 7, 7, 6, 7, 11, 9, 8, 10, 4, 11]# ваш код здесь
y = [28.1, 18.7, 1.0, 10.2, 11.6, 19.9, 24.4, 18.1, 18.5, 25.0, 21.8, 13.4, 18.0, 11.1, 21.1]# ваш код здесь

y_1 = [2 + 2*i for i in x] # раcсчитайте прогноз первой функцией
y_2 = [3 + 1*i for i in x] # раcсчитайте прогноз второй функцией

def error_function(y_real, y_pred):
    q =  sum((np.array(y_real) - np.array(y_pred)) ** 2 )/ len(y_real)
    return q

q_1 = error_function(y, y_1) # рассчитайте ошибку для прогноза первой модели
q_2 = error_function(y, y_2) # рассчитайте ошибку для прогноза второй модели

print('Ошибка первой модели:', q_1)
print('Ошибка второй модели:', q_2)
#######################################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# прочитайте данные с атрибутами аккаунтов компаний на Facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделяем данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# выведите название признаков в датасете
print(X.columns) # ваш код здесь

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# выведите среднее и стандартное отклонение признака 'Page total likes'
print('Mean for train', np.mean(X_train['Page total likes']))
print('Std for train', np.std(X_train['Page total likes']))

# стандартизируем данные 
scaler = StandardScaler()
scaler.fit(X_train) # обучите scaler на обучающей выборке методом fit
X_train_st = scaler.transform(X_train) # стандартизируйте обучающую выборку методом transform scaler 
X_test_st = scaler.transform(X_test) # стандартизируйте тестовую выборку методом transform scaler

print('Mean for standartized train', np.mean(X_train_st[:,0]))
print('Std for standartized train', np.std(X_train_st[:,0]))
print('Mean for standartized test', np.mean(X_test_st[:,0]))
print('Std for standartized test', np.std(X_test_st[:,0]))
########################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# прочитайте данные с атрибутами аккаунтов компаний на Facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделяем данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# стандартизируем данные методом StandartScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# зададим алгоритм для нашей модели
model = Lasso() # задайте модель как элемент класса Lasso

# обучим модель
model.fit(X_train_st, y_train) # обучите модель на стандартизированной обучающей выборке

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test_st) # сделайте прогноз для стандартизированной валидационной выборки 
#############################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# прочитаем данные с атрибутами аккаунтов компаний на Facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделяем данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# стандартизируем данные методом StandartScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# зададим алгоритм для нашей модели
model = Lasso()

# обучим модель
model.fit(X_train_st, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test_st)

# создадим датафрейм с признаками и их весами
features = pd.DataFrame({'feature':X.columns, 'coeff':model.coef_})# воспользуйтесь методом coef_
features['coeff_abs'] = abs(features['coeff']) # напишите свой код здесь


# выведите упорядоченный по модулю коэффициентов датафрейм с признаками
print(features.sort_values('coeff_abs', ascending=False))
##########################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# прочитаем данные с атрибутами аккаунтов компаний на Facebook и активностью на них
fb = pd.read_csv('/datasets/dataset_facebook_cosmetics.csv', sep = ';')

# разделяем данные на признаки (матрица X) и целевую переменную (y)
X = fb.drop('Total Interactions', axis = 1)
y = fb['Total Interactions']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# стандартизируем данные методом StandartScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# задаём алгоритм для нашей модели
model = Lasso()

# обучим модель
model.fit(X_train_st, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test_st)

# выведем среднее значение целевой переменной на тесте
print('Mean: {:.2f}'.format(y_test.mean()))

# выведем основные метрики
print('MAE: {:.2f}'.format(mean_absolute_error(y_test, predictions)))
print('MSE: {:.2f}'.format(mean_squared_error(y_test, predictions)))
print('R2: {:.2f}'.format(r2_score(y_test, predictions)))
##########################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# прочитайте из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv('/datasets/Electrical_Grid_Stability.csv', sep = ';') # прочитайте csv-файл и сохраните в переменной electrical_grid
print('Размер датасета:', electrical_grid.shape)
print(electrical_grid.head())

# разделите наши данные на признаки (матрица X) и целевую переменную (y)
#X = ...# сохраните в переменной матрицу объекты-признаки, удалив из датафрейма колонку с целевой переменной
#y = ...# сохраните в переменной колонку со значением целевой переменной

# разделяем данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis = 1)
y = electrical_grid['stability']


# разделите модель на обучающую и валидационную выборку
#X_train, X_test, y_train, y_test = ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#################################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# прочитайте из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv('Electrical_Grid_Stability.csv', sep = ';')
print('Размер датасета:', electrical_grid.shape)
print(electrical_grid.head())

# разделите наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis = 1)
y = electrical_grid['stability']

# разделите модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# задайте алгоритм для нашей модели
model = LogisticRegression()

# обучите модель
model.fit(X_train, y_train)

# воспользуйтесь уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test) #выдаёт вектор пар значений, где первое значение соответствует вероятности отнесения к первому ("0") классу, а второе — ко второму ("1")
probabilities = model.predict_proba(X_test)[:,1]

# выведите значения predictions и probabilities на экран
print(predictions)
print(probabilities)
###############################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# прочитаем из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv(
    '/datasets/Electrical_Grid_Stability.csv', sep=';'
)
print('Размер датасета:', electrical_grid.shape)
electrical_grid.head()

# посмотрим, как соотносятся классы набора данных
print('Соотношение классов:\n', electrical_grid['stability'].value_counts())

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis=1)
y = electrical_grid['stability']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# зададим алгоритм для нашей модели
model = LogisticRegression(solver='liblinear')

# обучим модель
model.fit(X_train, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
probabilities = model.predict_proba(X_test)[:, 1]

# бинарный прогноз
predictions = model.predict(X_test)

# выведите все изученные метрики для полученного прогноза
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, predictions)))
print('Recall: {:.2f}'.format(recall_score(y_test, predictions)))
print('F1: {:.2f}\n'.format(f1_score(y_test, predictions)))
#############################################################################################################
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

# прочитаем из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv(
    '/datasets/Electrical_Grid_Stability.csv', sep=';'
)
print('Размер датасета:', electrical_grid.shape)
electrical_grid.head()

# посмотрим, как соотносятся классы для нашего набора данных
electrical_grid['stability'].value_counts()

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis=1)
y = electrical_grid['stability']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# зададим алгоритм для нашей модели
model = LogisticRegression(solver='liblinear')

# обучим модель
model.fit(X_train, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
probabilities = model.predict_proba(X_test)[:,1]

# выведем roc_auc_score
print('ROC_AUC: {:.2f}'.format(roc_auc_score(y_test, probabilities)))
#####################################################################################################################
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# прочитаем из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv(
    '/datasets/Electrical_Grid_Stability.csv', sep=';'
)
print('Размер датасета:', electrical_grid.shape)
electrical_grid.head()

# посмотрим, как соотносятся классы набора данных
print('Соотношение классов:\n', electrical_grid['stability'].value_counts())

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis=1)
y = electrical_grid['stability']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# зададим алгоритм для нашей модели
model = LogisticRegression(solver='liblinear')

# обучим модель
model.fit(X_train, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
probabilities = model.predict_proba(X_test)[:, 1]

# бинарный прогноз
predictions = model.predict(X_test)

# выведите все изученные метрики для полученного прогноза
print('Метрики при автоматическом прогнозе с помощью predict')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, predictions)))
print('Recall: {:.2f}'.format(recall_score(y_test, predictions)))
print('F1: {:.2f}\n'.format(f1_score(y_test, predictions)))

# задайте порог
threshold = 0.4

# на основании вероятностей и соотношения классов рассчитайте predict
custom_predictions = [0 if i < threshold else 1 for i in probabilities]

# выведите все изученные метрики для прогноза по новому порогу
print('Метрики для прогноза с кастомным порогом')
print(
    'Accuracy for custom: {:.2f}'.format(
    accuracy_score(y_test, custom_predictions)
    )
)
print(
    'Precision for custom: {:.2f}'.format(
    precision_score(y_test, custom_predictions
    )
))
print(
    'Recall for custom: {:.2f}'.format(
    recall_score(y_test, custom_predictions
    )
))
print('F1 for custom: {:.2f}'.format(
    f1_score(y_test, custom_predictions
    )
))
###############################################################################################
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# прочитаем из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv(
    '/datasets/Electrical_Grid_Stability.csv', sep=';'
)
print('Размер датасета:', electrical_grid.shape)
electrical_grid.head()

# посмотрим, как соотносятся классы для нашего набора данных
electrical_grid['stability'].value_counts()

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis=1)
y = electrical_grid['stability']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# обучите StandartScaler на обучающей выборке
scaler = StandardScaler()
scaler.fit(X_train)

# Преобразуйте обучающий и валидационные наборы данных
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# зададим алгоритм для нашей модели
model = LogisticRegression(solver='liblinear', random_state=0)

# обучим модель
model.fit(X_train_st, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
predictions = model.predict(X_test_st)
probabilities = model.predict_proba(X_test_st)[:, 1]

# выведем все изученные метрики
print('Метрики для логистической регрессии')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, predictions)))
print('Recall: {:.2f}'.format(recall_score(y_test, predictions)))
print('F1: {:.2f}'.format(f1_score(y_test, predictions)))
print('ROC_AUC: {:.2f}\n'.format(roc_auc_score(y_test, probabilities)))

# зададим алгоритм для новой модели на основе алгоритма решающего дерева
tree_model = DecisionTreeClassifier(random_state=0)

# обучите модель
tree_model.fit(X_train_st, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
tree_predictions = tree_model.predict(X_test_st)
tree_probabilities = tree_model.predict_proba(X_test_st)[:,1]

# выведем все изученные метрики
print('Метрики для дерева принятия решения')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predictions)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predictions)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predictions)))
print('ROC_AUC: {:.2f}'.format(roc_auc_score(y_test, tree_predictions)))
#####################################################################################################
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# определим функцию, которая будет выводить наши метрики
def print_all_metrics(y_true, y_pred, y_proba, title='Метрики классификации'):
    print(title)
    print('\tAccuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
    print('\tPrecision: {:.2f}'.format(precision_score(y_true, y_pred)))
    print('\tRecall: {:.2f}'.format(recall_score(y_true, y_pred)))
    print('\tF1: {:.2f}'.format(f1_score(y_true, y_pred)))
    print('\tROC_AUC: {:.2f}'.format(roc_auc_score(y_true, y_proba)))

# прочитаем из csv-файла данные о параметрах сетей и их устойчивости
electrical_grid = pd.read_csv(
    '/datasets/Electrical_Grid_Stability.csv', sep=';'
)
print('Размер датасета:', electrical_grid.shape)
electrical_grid.head()

# посмотрим, как соотносятся классы для нашего набора данных
electrical_grid['stability'].value_counts()

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = electrical_grid.drop('stability', axis=1)
y = electrical_grid['stability']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# обучите StandartScaler на обучающей выборке
scaler = StandardScaler()
scaler.fit(X_train)

# Преобразуйте обучающий и валидационные наборы данных
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# зададим алгоритм для модели логистической регрессии
lr_model = LogisticRegression(random_state=0)
# обучим модель
lr_model.fit(X_train_st, y_train)
# воспользуемся уже обученной моделью, чтобы сделать прогнозы
lr_predictions = lr_model.predict(X_test_st)
lr_probabilities = lr_model.predict_proba(X_test_st)[:,1]
# выведем все метрики
print_all_metrics(
    y_test,
    lr_predictions,
    lr_probabilities,
    title='Метрики для модели логистической регрессии:',
)


# зададим алгоритм для новой модели на основе алгоритма решающего дерева
tree_model = DecisionTreeClassifier(random_state=0)
# обучим модель решающего дерева
tree_model.fit(X_train_st, y_train)
# воспользуемся уже обученной моделью, чтобы сделать прогнозы
tree_predictions = tree_model.predict(X_test_st)
tree_probabilities = tree_model.predict_proba(X_test_st)[:, 1]
# выведем все метрики
print_all_metrics(
    y_test,
    tree_predictions,
    tree_probabilities,
    title='Метрики для модели дерева решений:',
)


# зададим алгоритм для новой модели на основе алгоритма случайного леса
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 0) # Ваш код здесь
# обучим модель случайного леса
rf_model.fit(X_train_st, y_train)

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
rf_predictions = rf_model.predict(X_test_st) # Ваш код здесь
rf_probabilities = rf_model.predict_proba(X_test_st)[:,1] # Ваш код здесь
print_all_metrics(
        y_test, 
        rf_predictions, 
        rf_probabilities, 
        title = 'Метрики для модели случайного леса:')

# сделаем все то же самое для алгоритма градиентного бустинга
gb_model = GradientBoostingClassifier(n_estimators = 100, random_state = 0)
gb_model.fit(X_train_st, y_train)# обучим модель случайного леса

# воспользуемся уже обученной моделью, чтобы сделать прогнозы
gb_predictions = gb_model.predict(X_test_st)
gb_probabilities = gb_model.predict_proba(X_test_st)[:,1]

# выведем все метрики
print_all_metrics(
    y_test,
    gb_predictions,
    gb_probabilities,
    title='Метрики для модели градиентного бустинга:'
)

#########################################################################################################
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# определим функцию отрисовки графиков попарных признаков для кластеров
def show_clusters_on_plot(df, x_name, y_name, cluster_name):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        df[x_name], df[y_name], hue=df[cluster_name], palette='Paired'
    )
    plt.title('{} vs {}'.format(x_name, y_name))
    plt.show()


# читаем данные
travel = pd.read_csv('/datasets/tripadvisor_review_case.csv')
print(travel.shape)

# стандартизируем данные
sc = StandardScaler()
x_sc = sc.fit_transform(travel)

# задаём модель k_means с числом кластеров 3
km = KMeans(n_clusters = 3)

# прогнозируем кластеры для наблюдений (алгоритм присваивает им номера от 0 до 2)
labels = km.fit_predict(travel)

# сохраняем метки кластера в поле нашего датасета
travel['cluster_km'] = labels

# выводим статистику по средним значениям наших признаков по кластеру
print(travel.groupby('cluster_km').mean())

# отрисуем графики для пары признаков "соки" и "религия"
show_clusters_on_plot(travel, 
                            'Average user feedback on juice bars',
                            'Average user feedback on religious institutions',
                            'cluster_km') 

# отрисуем графики для пары признаков "соки" и "рестораны"
show_clusters_on_plot(travel,
                            'Average user feedback on juice bars',
                            'Average user feedback on restaurants',
                            'cluster_km')
#######################################################################################################
import pandas as pd
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# прочитаем данные
travel = pd.read_csv('/datasets/tripadvisor_review_case.csv')
print(travel.shape)

# стандартизируем данные
sc = StandardScaler()
x_sc = sc.fit_transform(travel)

# зададим модель k_means с количеством кластеров 3
km = KMeans(n_clusters = 3)
# спрогнозируем кластеры для наблюдений (алгоритм присваивает им номера от 0 до 2)
labels = km.fit_predict(x_sc)

# сохраним метки кластера в поле нашего датасета
travel['cluster_km'] = labels

# посчитаем метрику силуэта для нашей кластеризации
print('Silhouette_score: {:.2f}'.format(silhouette_score(x_sc, labels)))
##########################################################################################################
import pandas as pd

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())
#################################################################################################
import pandas as pd

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())

#посмотрим на сводную информацию о наборе данных
cars.info()
##################################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())

#посмотрим на сводную информацию о наборе данных
print(cars.info())

#построим и отрисуем матрицу корреляций
cm = cars.corr() #вычисляем матрицу корреляций
fig, ax = plt.subplots()

#нарисуем тепловую карту с подписями для матрицы корреляций
sns.heatmap(cm, square = True, annot = True) #ваш код здесь
ax.set_ylim(
    7, 0
)  # корректировка "рваных" полей heatmap в последней версии библиотеки
plt.show()

#построим попарные диаграммы рассеяния признак-целевая переменная для каждого признака
for col in cars.drop('Расход топлива', axis = 1).columns:
    sns.scatterplot(cars[col], cars['Расход топлива']) #ваш код здесь
    plt.show()
###############################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())

#уберите строки с пустыми значениями из набора данных и распечатайте размер датафрейма
cars.dropna(inplace = True)
print(cars.shape)
################################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())

#уберите строки с пустыми значениями из набора данных и распечатайте размер датафрейма
cars.dropna(inplace = True)
print(cars.shape)

#сохраним датафрейм с учётом преобразования признаков и распечатайте размер и первые 5 строк
cars = pd.get_dummies(cars)
print(cars.shape)
print(cars.head())
#####################################################################################################
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#прочитаем из csv-файла данные с описанием автомобилей и их расходом топлива
cars = pd.read_csv('/datasets/auto_cons.csv')

#распечатаем его размер и первые 5 строк
print(cars.shape)
print(cars.head())

#уберите строки с пустыми значениями из набора данных и распечатайте размер датафрейма
cars.dropna(inplace = True)
print(cars.shape)

#сохраним датафрейм с учётом преобразования признаков и распечатайте размер и первые 5 строк
cars = pd.get_dummies(cars)
print(cars.shape)
print(cars.head())

#разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = X = cars.drop('Расход топлива', axis =1) #ваш код здесь
y = cars['Расход топлива'] #ваш код здесь

#разделяем модель на обучающую и валидационную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #ваш код здесь

#создадим объект класса StandardScaler и применим его к обучающей выборке
scaler = StandardScaler() #ваш код здесь
X_train_st = scaler.fit_transform(X_train) #обучаем scaler и одновременно трансформируем матрицу для обучающей выборки
print(X_train_st[:5])

#применяем стандартизацию к матрице признаков для тестовой выборки
X_test_st = scaler.transform(X_test) #ваш код здесь
####################################################################################################
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# прочитаем из csv-файла данные с описанием автомобилей и их потреблением, распечатаем его размер и первые 5 строк
cars = pd.read_csv('/datasets/auto_cons.csv')

# уберём строки с пуcтыми значениями из набора данных
cars.dropna(inplace=True)

# сохраним датафрейм с учётом преобразования признаков
cars = pd.get_dummies(cars)

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = cars.drop(columns=['Расход топлива'])
y = cars['Расход топлива']

# разделяем модель на обучающую и валидационную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# создадим объект класса StandardScaler и применим его к обучающей выборке
scaler = StandardScaler()
X_train_st = scaler.fit_transform(
    X_train
)  # обучаем scaler и одновременно трансформируем матрицу для обучающей выборки

# применяем стандартизацию к матрице признаков для тестовой выборки
X_test_st = scaler.transform(X_test)

# задайте список моделей
models = [
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
]

# функция, которая вычисляет MAPE
def mape(y_true, y_pred):
    y_error = y_true - y_pred  # рассчитайте вектор ошибок
    y_error_abs = [abs(i) for i in y_error]  # рассчитайте вектор модуля ошибок
    perc_error_abs = y_error_abs / y_true  # рассчитайте вектор относительных ошибок
    mape = perc_error_abs.sum() / len(y_true)  # рассчитайте MAPE
    return mape


# функция, которая принимает на вход модель и данные и выводит метрики
def make_prediction(m, X_train, y_train, X_test, y_test):
    model = m
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) # ваш код здесь
    print('MAE:{:.2f} MSE:{:.2f} MAPE:{:.2f} R2:{:.2f} '\
        .format(mean_absolute_error(y_test,y_pred),
        mean_squared_error(y_test, y_pred),
        mape(y_test,y_pred),
        r2_score(y_test,y_pred)))
#######################################################################################################
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# прочитаем из csv-файла данные с описанием автомобилей и их потреблением, распечатаем его размер и первые 5 строк
cars = pd.read_csv('/datasets/auto_cons.csv')


# уберём строки с пустыми значениями из набора данных
cars.dropna(inplace=True)

# сохраним датафрейм с учётом преобразования признаков
cars = pd.get_dummies(cars)

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = cars.drop(columns=['Расход топлива'])
y = cars['Расход топлива']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# создадим объект класса StandardScaler и применим его к обучающей выборке
scaler = StandardScaler()
X_train_st = scaler.fit_transform(
    X_train
)  # обучаем scaler и одновременно трансформируем матрицу для обучающей выборки

# применяем стандартизацию к матрице признаков для тестовой выборки
X_test_st = scaler.transform(X_test)

# задайте список моделей
models = [
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
]

# функция, которая вычисляет MAPE
def mape(y_true, y_pred):
    y_error = y_true - y_pred
    y_error_abs = [abs(i) for i in y_error]
    perc_error_abs = y_error_abs / y_true
    mape = perc_error_abs.sum() / len(y_true)
    return mape


# функция, которая принимает на вход модель и данные и выводит метрики
def make_prediction(m, X_train, y_train, X_test, y_test):
    model = m
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        'MAE:{:.2f} MSE:{:.2f} MAPE:{:.2f} R2:{:.2f} '.format(
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            mape(y_test, y_pred),
            r2_score(y_test, y_pred),
        )
    )


# напишите цикл, который выводит метрики по списку моделей
for i in models:
    print(i)
    make_prediction(m=i,X_train = X_train_st,
                                    y_train= y_train,
                                    X_test=X_test_st,
                                    y_test = y_test)
######################################################################################################
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# прочитаем из csv-файла данные с описанием автомобилей и их потреблением, распечатаем его размер и первые 5 строк
cars = pd.read_csv('/datasets/auto_cons.csv')

# уберём строки с пустыми значениями из набора данных
cars.dropna(inplace=True)

# сохраним датафрейм с учётом преобразования признаков
cars = pd.get_dummies(cars)

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = cars.drop(columns=['Расход топлива'])
y = cars['Расход топлива']

# разделяем модель на обучающую и валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# создадим объект класса StandardScaler и применим его к обучающей выборке
scaler = StandardScaler()
X_train_st = scaler.fit_transform(
    X_train
)  # обучаем scaler и одновременно трансформируем матрицу для обучающей выборки

# применяем стандартизацию к матрице признаков для тестовой выборки
X_test_st = scaler.transform(X_test)

# задайте список моделей
models = [
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
]

# функция, которая вычисляет MAPE
def mape(y_true, y_pred):
    y_error = y_true - y_pred
    y_error_abs = [abs(i) for i in y_error]
    perc_error_abs = y_error_abs / y_true
    return perc_error_abs.sum() / len(y_true)


# функция, которая принимает на вход модель и данные и выводит метрики
def make_prediction(m, X_train, y_train, X_test, y_test):
    model = m
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        'MAE:{:.2f} MSE:{:.2f} MAPE:{:.2f} R2:{:.2f} '.format(
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            mape(y_test, y_pred),
            r2_score(y_test, y_pred),
        )
    )


# напишите цикл, который выводит метрики по списку моделей
for i in models:
    print(i)
    make_prediction(
        m=i,
        X_train=X_train_st,
        y_train=y_train,
        X_test=X_test_st,
        y_test=y_test,
    )

# обучим финальную модель
final_model = GradientBoostingRegressor()
final_model.fit(X_train_st, y_train)
y_pred = final_model.predict(X_test_st)

# создадим датафрейм с именами признаков и их важностью и выведем его по убыванию важности
importance = pd.DataFrame(data={'feature': X.columns, 'importance': final_model.feature_importances_})\
                                                        .sort_values(by='importance', ascending=False)
print(importance)








































