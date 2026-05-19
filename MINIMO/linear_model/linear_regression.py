import numpy as np
from MINIMO.base import BaseModel

class linear_regression(BaseModel):
    def __init__(self,lr = 0.01,epoch=1000):
        self.lr = lr
        self.epoch = epoch
      
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias =0

        for _ in range(self.epoch):
            y_pred = np.dot(X,self.weights) +self.bias

            dw = (1/n_samples) *np.dot(X.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias