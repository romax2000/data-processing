import sys
print("версия Python: {}".format(sys.version))

import pandas as pd
print("версия pandas: {}".format(pd.__version__))

import matplotlib
print("версия matplotlib: {}".format(matplotlib.__version__))

import numpy as np
print("версия NumPy: {}".format(np.__version__))

import scipy as sp
print("версия SciPy: {}".format(sp.__version__))

import IPython
print("версия IPython: {}".format(IPython.__version__))

import sklearn
print("версия scikit-learn: {}".format(sklearn.__version__))
print("#####################################")


from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия ответов: {}".format(iris_dataset['target_names']))
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape)) #150 цветов по 4-ём различным признакам
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
print("Тип массива target: {}".format(type(iris_dataset['target']))) #сорта уже измеренных цветов
print("Форма массива target: {}".format(iris_dataset['target'].shape)) #target представляет собой одномерный массив, по одному элементу для каждого цветка
print("Ответы: \n{}".format(iris_dataset['target'])) #Значения чисел задаются массивом iris['target_names']: 0 – setosa, 1 – versicolor, а 2 – virginica.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
#train_test_split перемешивает набор данных с помощью генератора псевдослучайных чисел. 
print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))

print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

#	создаем dataframe из данных в массиве X_train
#	маркируем столбцы, используя строки в iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 	создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train 
#Способ в jupyter
from pandas.plotting import scatter_matrix
import mglearn
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
iris_dataframe.info()

'''
#Способ с окошком
import matplotlib.pyplot as plt
import mglearn
from pandas.plotting import scatter_matrix

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

X_train,X_test,Y_train,Y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

grr = scatter_matrix(iris_dataframe,c = Y_train,figsize = (15,15),marker = 'o',
hist_kwds={'bins':20},s=60,alpha=.8,cmap = mglearn.cm3)
plt.show()
'''

print("222222222222222222222222222222222222222222222222")

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)
'''
включает в себя алгоритм, 
который будет использоваться
для построения модели на обучающих данных, 
а также алгоритм, который сгенерирует прогнозы для 
новых точек данных. Он также будет содержать информацию, 
которую алгоритм извлек из обучающих данных.
В случае с KNeighborsClassifier он будет просто хранить
обучающий набор.
'''
knn.fit(X_train, y_train) #Метод fit возвращает сам объект knn (и изменяет его), таким образом, мы получаем строковое представление нашего классификатора. Оно показывает нам, какие параметры были использованы при создании модели
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

prediction = knn.predict(X_new) #прогноз
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(
iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))

print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))


