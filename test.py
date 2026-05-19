import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
from MINIMO.linear_model.linear_regression import linear_regression
from MINIMO.linear_model.logistic_regression import logistic_regression

from sklearn.linear_model import LogisticRegression

a = pd.read_csv("student_dataset_detailed.csv")

a=a.drop(columns=['Name'])
a=a.drop(columns=['Department'])
a=a.drop(columns=['Extracurricular'])

# a = a.dropna()

a['Age']=a['Age'].fillna(a['Age'].mode()[0])
a['Gender']=a['Gender'].fillna(a['Gender'].mode()[0])
a['Study_Hours']=a['Study_Hours'].fillna(a['Study_Hours'].mode()[0])
a['Attendance']=a['Attendance'].fillna(a['Attendance'].mode()[0])
a['Assignments_Submitted']=a['Assignments_Submitted'].fillna(a['Assignments_Submitted'].mode()[0])
a['Previous_Score']=a['Previous_Score'].fillna(a['Previous_Score'].mode()[0])
a['Score']=a['Score'].fillna(a['Score'].mode()[0])
a['Result']=a['Result'].fillna(a['Result'].mode()[0])


en = LabelEncoder()

a['Gender'] = en.fit_transform(a['Gender'])
a['Result'] = en.fit_transform(a['Result'])

X = a.drop(columns='Score')
y = a['Score']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

scaler = MinMaxScaler()
s = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = linear_regression()


model.fit(X_train,y_train)

c = model.predict(X_test)

mse = mean_squared_error(y_test, c)
root=np.sqrt(mse)

print(c)

print("mean squared error are : ",mse)
print("root squared error are : ",root)


# logistic 
X = a.drop(columns='Result')
y = a['Result']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values


X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

m = logistic_regression()

n = LogisticRegression()

m.fit(X_train,y_train)
n.fit(X_train,y_train)

d = m.predict(X_test)
f = n.predict(X_test)

print(d)
print(f)

acc = accuracy_score(y_test, d)
ac = accuracy_score(y_test, f)

print("Accuracy:", acc)
print("Accuracy of real :", ac)

cm = confusion_matrix(y_test, d)
cma = confusion_matrix(y_test, f)

print("confusion metrics ",cm)
print("confusion metrics ",cma)
