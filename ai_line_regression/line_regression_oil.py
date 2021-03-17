import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import time

# read data in xlsx
gpd =  pd.read_excel('gdprussia.xlsx') 

print(gpd)
# output data 
# plt.scatter( gpd.oilprice ,  gpd.gdp  )
# plt.xlabel('oil price (US$)')
# plt.ylabel('GDP Rashka (bln US$)')
# show graph
# plt.show()

# trening model

# build model
reg = linear_model.LinearRegression()

# # # learn model
reg.fit(gpd[[ 'year' , 'oilprice']], gpd.gdp )

prprpr = reg.predict(gpd[[ 'year' , 'oilprice']])

# prediction year - [ price, oil ]
qqqq = reg.predict([[2025, 10]])
print(qqqq) 
wwww = reg.predict([[2025, 5]])
print(wwww) 

