import sklearn.naive_bayes as naive_bayes
import numpy as np

class nb():

    def train_nb(self, train_X, train_Y):
        model = naive_bayes.GaussianNB()
        model.fit(train_X, train_Y)
        return model

    def test_nb(self, model, test_X):
        pred = model.predict_proba(test_X)
        return np.array(pred[:,1])