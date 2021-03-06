
import pandas as pd
import numpy as np


#  take data in kaggle.com/c/titanic/data
data = pd.read_csv('titanic.csv')


columns_target = ['Survived'] # наша целевая колонка

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']



X = data[columns_train]
Y = data[columns_target]


# Проверяем есть ли пустые ячейки в колонках
X['Sex'].isnull().sum()

X['Pclass'].isnull().sum()

X['Fare'].isnull().sum()

X['Age'].isnull().sum()

# Заполняем пустые ячейки медианным значением по возрасту

X['Age'] = X['Age'].fillna(X['Age'].mean())

X['Age'].isnull().sum()


# In[192]:


# Заменяем male и female на 0 и 1 с помощью словаря


# In[193]:


d={'male':0, 'female':1} # создаем словарь


# In[194]:


X['Sex'] = X['Sex'].apply(lambda x:d[x])


# In[195]:


X['Sex'].head() 


# In[196]:


# Разделяем нашу выборку на обучающую и тестовую


# In[197]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


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




