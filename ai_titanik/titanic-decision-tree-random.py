import pandas as pd
import numpy as np
import eli5

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)  # test_size=0.1 - уменьшаем или увеличиваем размер тестовой выборки

model = RandomForestClassifier(n_estimators=100)  # обучаем модель с помощью случайного леса

model.fit(X_train,Y_train)   # тренируем модель

res = model.score(X_test, Y_test)  # смотрим результат 
print(res)

















