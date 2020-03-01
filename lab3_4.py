import matplotlib.pyplot as plt
import numpy as np


X, y = mglearn.datasets.make_forge()
# строим график для набора данных
%matplotlib inline
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
#Отдельно в юпитере
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
#Отдельно
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Ключи cancer(): \n{}".format(cancer.keys()))
print("Форма массива data для набора cancer: {}".format(cancer.data.shape))
print("Количество примеров для каждого класса:\n{}".format(
{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Имена признаков:\n{}".format(cancer.feature_names))
from sklearn.datasets import load_boston
boston = load_boston()
print("форма массива data для набора boston: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()#набор данных с производными признаками
print("форма массива X: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=1)#ближайшие соседи
mglearn.plots.plot_knn_classification(n_neighbors=3)

# разделение на тестовый и обучающий наборы
from sklearn.model_selection import train_test_split 
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
#создаем объект-классификатор и подгоняем в одной строке
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title("количество соседей:{}".format(n_neighbors)) #не подписаны графики
	ax.set_xlabel("признак 0")
	ax.set_ylabel("признак 1")
axes[0].legend(loc=3)

##########классификация ближайшими соседями

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

#пробуем n_neighbors от 1 до 10 
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
#строим модель
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)

#записываем правильность на обучающем наборе 
	training_accuracy.append(clf.score(X_train, y_train))

#записываем правильность на тестовом наборе
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()

##########регрессия ближайшими соседями

mglearn.plots.plot_knn_regression(n_neighbors=1)# это целевое значение ближайшего соседа
mglearn.plots.plot_knn_regression(n_neighbors=3)# среднее значение соответствующих соседей 


from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
#разбиваем набор данных wave на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#создаем экземпляр модели и устанавливаем количество соседей равным 3 
reg = KNeighborsRegressor(n_neighbors=3)
#подгоняем модель с использованием обучающих данных и обучающих ответов 
reg.fit(X_train, y_train)
print("Прогнозы для тестового набора:\n{}".format(reg.predict(X_test)))
print("R^2 на тестовом наборе: {:.2f}".format(reg.score(X_test, y_test)))

########

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#создаем 1000 точек данных, равномерно распределенных между -3 и 3 
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
#получаем прогнозы, используя 1, 3, и 9 соседей
	reg = KNeighborsRegressor(n_neighbors=n_neighbors) 
	reg.fit(X_train, y_train)
	ax.plot(line, reg.predict(line))
	ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8) 
	ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
	ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format( n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
	ax.set_xlabel("Признак")
	ax.set_ylabel("Целевая переменная")
axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы",
"Тестовые данные/ответы"], loc="best")


#lab4
mglearn.plots.plot_linear_regression_wave()

#Линейная регрессия или обычный метод наименьших квадратов 
'''
Линейная регрессия находит параметры w и b, 
которые минимизируют среднеквадратическую ошибку 
(mean squared error) между спрогнозированными 
и фактическими ответами у в обучающем наборе. 
Среднеквадратичная ошибка равна сумме квадратов разностей 
между спрогнозированными и фактическими значениями.
'''
from sklearn.linear_model import LinearRegression 
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

#Параметры «наклона» (w) в атрибуте coef_,
#тогда как сдвиг (offset), обозначаемая как b в атрибуте intercept_:
print("lr.coef_: {}".format(lr.coef_)) #coef_ - это массив NumPy, в котором каждому элементу соответствует входной признак.
print("lr.intercept_: {}".format(lr.intercept_)) #intercept_ - это всегда отдельное число с плавающей точкой

print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test)))

#
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

#явным признаком переобучения
print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test)))

#гребневая регрессия
'''
В гребневой регрессии коэффициенты (w)
 выбираются не только с точки зрения того, 
 насколько хорошо они позволяют предсказывать 
 на обучающих данных, они еще подгоняются в 
 соответствии с дополнительным ограничением.
  Нам нужно, чтобы величина коэффициентов была как
 можно меньше. Другими словами, все элементы w
  должны быть близки к нулю. Это означает, 
  что каждый признак должен иметь как можно 
  меньшее влияние на результат 
'''

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(ridge.score(X_test, y_test)))

'''
Увеличение alpha заставляет коэффициенты сжиматься до близких к нулю значений, 
что снижает качество работы модели на обучающем наборе,
но может улучшить ее обобщающую способность. 
'''
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(ridge10.score(X_test, y_test)))

#При очень малых значениях alpha, ограничение на коэффициенты практически не накладывается 
#и мы в конечном итоге получаем модель, напоминающую линейную регрессию
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(ridge01.score(X_test, y_test)))

#демонстрация вышесказанного
plt.plot(ridge.coef_, 's', label="Гребневая регрессия alpha=1")
plt.plot(ridge10.coef_, '^', label="Гребневая регрессия alpha=10")
plt.plot(ridge01.coef_, 'v', label="Гребневая регрессия alpha=0.1")

plt.plot(lr.coef_, 'o', label="Линейная регрессия")
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

'''
независимо от объема данных правильность на обучающем наборе
 всегда выше правильности на тестовом наборе, как в 
 случае использования гребневой регрессии, так и в 
 случае использования линейной регрессии. Поскольку 
 гребневая регрессия – регуляризированная модель, во 
 всех случаях на обучающем наборе правильность гребневой
  регрессии ниже правильности линейной регрессии.
 Однако правильность на тестовом наборе у гребневой 
 регрессии выше, особенно для небольших подмножеств данных. 
'''
mglearn.plots.plot_ridge_n_samples()

'''
По мере возрастания объема данных, 
доступного для моделирования, обе модели становятся
 лучше и в итоге линейная регрессия догоняет 
 гребневую регрессию. Урок здесь состоит в том, 
 что при достаточном объеме обучающих данных
 регуляризация становится менее важной и при
 удовлетворительном объеме данных гребневая и 
 линейная регрессии будут демонстрировать одинаковое качество работы
'''

#снижение правильности линейной регрессии на обучающем 
#наборе c возрастанием объема данных модели становится 
#все сложнее переобучиться или запомнить данные.

#Лассо

#Как и гребневая регрессия, лассо также сжимает коэффициенты до близких к нулю значений, но несколько иным способом

'''
Получается, что некоторые признаки полностью исключаются из модели. Это можно рассматривать как один из видов автоматического отбора признаков. Получение нулевых значений для 
некоторых коэффициентов часто упрощает интерпретацию модели и может выявить наиболее важные признаки вашей модели.
'''

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(lasso.score(X_train, y_train))) #недообучение
print("Правильность на контрольном наборе: {:.2f}".format(lasso.score(X_test, y_test)))
print("Количество использованных признаков: {}".format(np.sum(lasso.coef_ != 0)))


lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Количество использованных признаков: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Количество использованных признаков: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso.coef_, 's', label="Лассо alpha=1")
plt.plot(lasso001.coef_, '^', label="Лассо alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Лассо alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Гребневая регрессия alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")


#lab5


