import numpy as np
from MINIMO.base import BaseModel

class logistic_regression(BaseModel):
    def __init__(self,lr = 0.001,epoch=10):
        self.lr = lr
        self.epoch = epoch

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias =0

        for _ in range(self.epoch):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/n_samples) *np.dot(X.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return np.where(y_prob >= 0.5, 1, 0)
    # def predict(self,X):
    #     return np.dot(X,self.weights) + self.bias