class BaseModel():
    def fit(self,x,y):
        raise NotImplementedError
    def predict(self,x):
        raise NotImplementedError
