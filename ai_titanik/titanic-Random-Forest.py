#!/usr/bin/env python
# coding: utf-8

# In[112]:


# Импортируем модули


# In[2]:


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# In[3]:


# загружаем данные


# In[4]:


data = pd.read_csv('titanic.csv')


# In[5]:


data


# In[6]:


# Предварительная работа с данными 


# In[7]:


columns_target = ['Survived'] # наша целевая колонка

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']


# In[8]:


X = data[columns_train]
Y = data[columns_target]


# In[9]:


# Проверяем есть ли пустые ячейки в колонках


# In[10]:


X['Sex'].isnull().sum()


# In[38]:


X['Pclass'].isnull().sum()


# In[39]:


X['Fare'].isnull().sum()


# In[40]:


X['Age'].isnull().sum()


# In[41]:


# Заполняем пустые ячейки медианным значением по возрасту


# In[14]:


pd.options.mode.chained_assignment = None # отключаем розовые предупреждения)


# In[15]:


X['Age'] = X['Age'].fillna(X['Age'].mean())


# In[16]:


X['Age'].isnull().sum()


# In[17]:


# Заменяем male и female на 0 и 1 с помощью словаря


# In[18]:


d={'male':0, 'female':1} # создаем словарь


# In[19]:


X['Sex'] = X['Sex'].apply(lambda x:d[x])


# In[20]:


X['Sex'].head() 


# In[21]:


# Разделяем нашу выборку на обучающую и тестовую


# In[35]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[36]:


model = RandomForestClassifier(n_estimators=100)


# In[37]:


model.fit(X_train,Y_train)


# In[38]:


model.score(X_test,Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[198]:


# Загружаем модель Support VEctor Machine для обучения


# In[199]:


from sklearn import svm


# In[200]:


predmodel = svm.LinearSVC()


# In[201]:


# Обучаем модель с помощью нашей обучающей выборки


# In[202]:


predmodel.fit(X_train, Y_train)


# In[203]:


# Предсказываем на тестовой выборке


# In[204]:


predmodel.predict(X_test[0:10])


# In[205]:


# Проверяем точность предсказаний


# In[206]:


predmodel.score(X_test,Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




