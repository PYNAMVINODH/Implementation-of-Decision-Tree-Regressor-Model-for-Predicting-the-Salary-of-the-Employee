# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. importing libraries
2. reading data set
3. using label encoder
4. train and test the model
5. finging mse and metrics_score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PYNAM VINODH
RegisterNumber:  212223240131
import pandas as pd
data=pd.read_csv('Salary.csv')
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
dt.predict([[5,6]])

*/
```

## Output:
## Data:
![image](https://github.com/PYNAMVINODH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742678/50e26e71-467e-48f5-8576-89ead894d9ca)

## Label:
![image](https://github.com/PYNAMVINODH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742678/13a62c16-2110-49fd-a52e-bfcc24b449b8)

## Mean Square Error:
![image](https://github.com/PYNAMVINODH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742678/d703782d-4538-4446-b783-65ae806ffb99)

## Metrics Score:
![image](https://github.com/PYNAMVINODH/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742678/ccac3ea3-993c-41ee-93ec-b65077dcd19a)








## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
