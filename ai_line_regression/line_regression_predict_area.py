import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import time

# read data in xlsx
df =  pd.read_excel('price1.xlsx') 

# output data 
plt.scatter( df.area, df.price )
plt.xlabel('площадь м2')
plt.ylabel('money')
# show graph
# plt.show()

# trening model

# build model
reg = linear_model.LinearRegression()

# learn model
reg.fit(df[['area']], df.price)

# prediction
qqqq = reg.predict([[318]])
print(qqqq) 
wwww = reg.predict([[31]])
print(wwww) 
prprpr = reg.predict(df[['area']])
print(prprpr) 

coef = reg.coef_
print(coef) 

inter = reg.intercept_
print(inter) 


# predict model graph 
plt.scatter( df.area, df.price )
plt.xlabel('площадь м2')
plt.ylabel('money')
plt.plot(df.area, reg.predict(df[['area']]))
# show graph
# plt.show()

pred =  pd.read_excel('prediction_price.xlsx') 
predicted_prices = reg.predict(pred)

pred['predicted_prices'] = predicted_prices

print(pred) 
pred.to_excel('predicted_prices.xlsx')