import pandas as pd
import numpy as np
import eli5

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#  take data in kaggle.com/c/titanic/data
data = pd.read_csv('train.csv')

# print(data)

columns_target = ['Survived'] # наша целевая колонка

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']


X = data[columns_train]
Y = data[columns_target]


# Проверяем есть ли пустые ячейки в колонках
X['Sex'].isnull().sum()

X['Pclass'].isnull().sum()

X['Fare'].isnull().sum()

X['Age'].isnull().sum()

#  Заполняем пустые ячейки медианным значением по возрасту
pd.options.mode.chained_assignment = None # warning off

X['Age'] = X['Age'].fillna(X['Age'].mean())

X['Age'].isnull().sum()


# # Заменяем male и female на 0 и 1 с помощью словаря

d={'male':0, 'female':1} # создаем словарь
pd.options.mode.chained_assignment = None # warning off

X['Sex'] = X['Sex'].apply(lambda x:d[x])

# print(X['Sex'].head())

# # Разделяем нашу выборку на обучающую и тестовую

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21) # строим дерево
clf.fit(X_train, Y_train)      # обучаем модель
aaa =  clf.score(X_test, Y_test)    # проверяем точность
print(aaa)


# eli5.explain_weigths_sklearn( clf, feature_names=X_train.columns.values)

# # Загружаем модель Support VEctor Machine для обучения

# from sklearn import svm

# predmodel = svm.LinearSVC()

# # # Обучаем модель с помощью нашей обучающей выборки

# predmodel.fit(X_train, Y_train)

# # # Предсказываем на тестовой выборке

# pred1 = predmodel.predict(X_test[0:10])
# print(pred1)
# # # Проверяем точность предсказаний

# pred1Score = predmodel.score(X_test,Y_test)

# print(pred1Score)





