import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv('Salary_Data.csv')

meanofdata= data['Salary'].mean()
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=10)


from sklearn.linear_model import LinearRegression  #importing LinearRegression class
lr_clf=LinearRegression()             #creating object of the class
lr_clf.fit(X_train, y_train)          #using fit method of LinearRegression class
lr_clf.score(X_test, y_test)
 


pred_salary= lr_clf.predict(X_test)


rmse = mean_squared_error(y_test, pred_salary)  

#data.plot.scatter(x='YearsExperience', y='Salary')


'''plt.scatter(data.YearsExperience, data.Salary)
plt.plot(data.Salary, meanofdata, color='orange')'''


plt.scatter(X_train,y_train, color= 'red')
plt.plot(X_train,lr_clf.predict(X_train), color='orange')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()


plt.scatter(X_test,y_test, color= 'red')
plt.plot(X_test,lr_clf.predict(X_test), color='orange')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()




fromuser= int(input())
lr_clf.predict([[fromuser]])



print(rmse)





