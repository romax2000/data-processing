from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
#метод опорных векторов и логистическая регрессия
X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
ax=ax, alpha=.7)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title("{}".format(clf.__class__.__name__))
	ax.set_xlabel("Признак 0")
	ax.set_ylabel("Признак 1")
axes[0].legend()

mglearn.plots.plot_linear_svc_regularization()
#C - степень регуляризации
#низкий С - высокая подстраиваемость
#высокий С - пытается подчеркнуть важность каждой точки

#разбор логистической регрессии
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg.score(X_test, y_test)))
#вроде модель недоучена


logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg100.score(X_test, y_test)))
#увеличили С и получили более высокую правильность


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg001.score(X_test, y_test)))
#уменьшили


plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.legend()


#Если мы хотим получить более интерпретабельную модель, нам может помочь L1 регуляризация,
#поскольку она ограничивает модель использованием лишь нескольких признаков. 

#сравнение с l1
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
	lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)#проблема с l1, работает с l2
	print("Правильность на обучении для логрегрессии l1 с C={:.3f}: {:.2f}".format( C, lr_l1.score(X_train, y_train)))
	print("Правильность на тесте для логрегрессии l1 с C={:.3f}: {:.2f}".format( C, lr_l1.score(X_test, y_test)))
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")

plt.ylim(-5, 5)
plt.legend(loc=3)

#6лаба мультиклассовая классификация
'''
В подходе «один против остальных» для каждого класса строится бинарная модель, которая пытается отделить
этот класс от всех остальных, в результате чего количество моделей определяется количеством классов.
'''

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.legend(["Класс 0", "Класс 1", "Класс 2"])


#Теперь обучаем классификатор LinearSVC на этом наборе данных:
linear_svm = LinearSVC().fit(X, y)
print("Форма коэффициента: ", linear_svm.coef_.shape)
print("Форма константы: ", linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.legend(['Класс 0', 'Класс 1', 'Класс 2', 'Линия класса 0', 'Линия класса 1', 'Линия класса 2'], loc=(1.01, 0.3))

'''
Какой класс будет присвоен точке, расположенной в треугольнике? Ответ – класс, получивший наибольшее значение по формуле классификации, 
то есть класс ближайшей линии.
'''

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Класс 0', 'Класс 1', 'Класс 2', 'Линия класса 0', 'Линия класса 1',
'Линия класса 2'], loc=(1.01, 0.3))
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")

#Большие значения alpha или маленькие значения C означают простые модели. 
#Если вы полагаете, что на самом деле важны лишь некоторые признаки, следует использовать L1
#Еще L1 регуляризация может быть полезна, если интерпретируемость модели имеет важное значение.
#линейные модели хорошо работают, когда количество признаков превышает количество наблюдений


#цепочка методов
logreg = LogisticRegression().fit(X_train, y_train)#	создаем экземпляр модели и подгоняем его в одной строке 
logreg = LogisticRegression()
y_pred = logreg.fit(X_train, y_train).predict(X_test)# это связывание методов fit и predict в одной строке
y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test) # создать экземпляр модели, подогнать модель и получить прогнозы в одной строке

#lab7
#Наивные байесовские классификаторы - обучаются быстрее но более низкая обобщающая способность

'''
они оценивают параметры, рассматривая каждый признак отдельно и по каждому 
признаку собирают простые статистики классов
'''

#Классификатор BernoulliNB - принимает бинарные данные
X = np.array([[0, 1, 0, 1],
 [1, 0, 1, 1],
 [0, 0, 0, 1],
 [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
	#итерируем по каждому классу
	#подсчитываем (суммируем) элементы 1 по признаку
	counts[label] = X[y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))

#MultinomialNB и BernoulliNB имеют один параметр alpha, который контролирует сложность модели
'''
алгоритм добавляет к данным зависящее от alpha
 определенное количество искусственных наблюдений
  с положительными значениями для всех признаков. 
  Это приводит к «сглаживанию» статистик. Большее 
  значение alpha означает более высокую степень 
  сглаживания, что приводит к построению менее сложных 
   моделей. 
'''
#тонкая настройка этого параметра обычно немного увеличивает правильность
#GaussianNB в основном используется для данных с очень высокой размерностью
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
mglearn.plots.plot_animal_tree()

'''
Построение дерева решений означает построение
 последовательности правил «если… то…», которая приводит
 нас к истинному ответу максимально коротким путем. 
 В машинном обучении эти правила называются тестами (tests)
'''
mglearn.plots.plot_tree_progressive() #пример рекурсивного разбиения


'''
сначала выясняют, в какой области разбиения пространства
 признаков находится данная точка, а затем определяют
  класс, к которому относится большинство точек
   в этой области (либо единственный класс в области,
    если лист является чистым). Область может быть 
    найдена с помощью обхода дерева, начиная с корневого
    узла и путем перемещения влево или вправо, 
    в зависимости от того, выполняется ли тест или нет
'''

#предварительная обрезка
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))
#Дерево имеет глубину, как раз достаточную для того, чтобы прекрасно запомнить все метки обучающих данных.

tree = DecisionTreeClassifier(max_depth=4, random_state=0)#ограничиваем глубину
tree.fit(X_train, y_train)

print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


'''
вариант для pdf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
%matplotlib inline

from sklearn.model_selection import train_test_split from sklearn.datasets import load_breast_cancer from sklearn import tree
from sklearn.tree import export_graphviz
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=0)
clf = clf.fit(X_train, y_train)

import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("cancer.pdf")


'''

'''
png

from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=cancer.feature_names,
class_names=cancer.target_names,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

'''


#важности признака

print("Важности признаков:\n{}".format(tree.feature_importances_))
for name, score in zip(cancer["feature_names"], tree.feature_importances_):
	print(name, score)

def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Важность признака")
	plt.ylabel("Признак")

plot_feature_importances_cancer(tree)


tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

#регрессия деревья
'''
не умеет экстраполировать или делать 
прогнозы вне диапазона значений обучающих данных.
'''

import pandas as pd
ram_prices = pd.read_csv("C:/Data/ram_price.csv")

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Год")
plt.ylabel("Цена $/Мбайт")


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
#используем исторические данные для прогнозирования цен после 2000 года 
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
# прогнозируем цены по датам
X_train = data_train.date[:, np.newaxis]
# мы используем логпреобразование, что получить простую взаимосвязь между данными и откликом
y_train = np.log(data_train.price)
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
# прогнозируем по всем данным
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
# экспоненцируем, чтобы обратить логарифмическое преобразование
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)


#Сравнение прогнозов линейной модели и прогнозов дерева регрессии для набора данных RAM price
plt.semilogy(data_train.date, data_train.price, label="Обучающие данные")
plt.semilogy(data_test.date, data_test.price, label="Тестовые данные")
plt.semilogy(ram_prices.date, price_tree, label="Прогнозы дерева")
plt.semilogy(ram_prices.date, price_lr, label="Прогнозы линейной регрессии")
plt.legend()

'''
Дерево не способно генерировать «новые» ответы, 
выходящие за пределы значений обучающих данных. Этот недостаток 
относится ко всем моделям на основе деревьев решений
'''


#lab8

#Случайный лес(рандом форест)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2) #кол-во деревьев
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
	ax.set_title("Дерево {}".format(i))
	mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

	mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
alpha=.4)
axes[-1, -1].set_title("Случайный лес")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
 


X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(forest.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(forest.score(X_test, y_test)))


def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Важность признака")
	plt.ylabel("Признак")
plot_feature_importances_cancer(forest)
#Однако обратная сторона увеличения числа деревьев заключается в том, 
#что с ростом количества деревьев требуется больше памяти и больше времени для обучения.



#градиентный бустинг деревьев регрессии
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(gbrt.score(X_test, y_test)))


gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(gbrt.score(X_test, y_test)))


gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(gbrt.score(X_test, y_test)))


gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Важность признака")
	plt.ylabel("Признак")
plot_feature_importances_cancer(gbrt)




#lab9
