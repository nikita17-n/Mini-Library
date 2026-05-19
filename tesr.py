import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from MINIMO.linear_model.linear_regression import linear_regression
from sklearn.preprocessing import MinMaxScaler

a = pd.read_csv("student_dataset_detailed.csv")

a=a.drop(columns=['Name'])
a=a.drop(columns=['Department'])
a=a.drop(columns=['Extracurricular'])


print(a.isnull().sum())
a = a.dropna()
en = LabelEncoder()

a['Gender'] = en.fit_transform(a['Gender'])
a['Result'] = en.fit_transform(a['Result'])

X = a.drop(columns='Score')
y = a['Score']

s = MinMaxScaler()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

m = LinearRegression()
m.fit(X_train,y_train)
d = m.predict(X_test)

model = linear_regression()
model.fit(X_train,y_train)

c = model.predict(X_test)
e = model.predict(X_train)



print("mse: ",np.sqrt(mean_squared_error(y_test,c)))
print("mse: ",np.sqrt(mean_squared_error(y_test,d)))
print("mse: ",np.sqrt(mean_squared_error(y_train,e)))


print(c)
print(d) 

