import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score


data= pd.read_csv('Position_Salaries.csv')


plt.scatter(data.Level, data.Salary)    #datasets are making a curve, make a linear model non-suitble

X=data.iloc[:, 1:2].values
y=data.iloc[:,-1].values
     
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_reg= poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_pol=LinearRegression()
lin_pol_fit= lin_pol.fit(X_reg,y)
#lin_pol.score(X_reg, y)
 

plt.scatter(X,y, color= 'red')
X_grid= np.arange(min(X), max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.plot(X_grid,lin_pol_fit.predict(poly_reg.fit_transform(X_grid)), color='orange')
plt.title('S vs E') 
plt.xlabel('Position Label')
plt.ylabel('salay')
plt.show()
 
 #comparing with linear regression

lin_reg1= LinearRegression()    
l_fit=lin_reg1.fit(X, y)
#lin_reg1.score(X, y)


plt.scatter(X,y, color= 'red')
plt.plot(X, l_fit.predict(X), color= 'red')

plt.title('Salary vs Experience')
plt.xlabel('Position Label')
plt.ylabel('salary')
plt.show()

#r2 value in both the models

lin_reg1.score(X, y)
lin_pol.score(X_reg, y)





