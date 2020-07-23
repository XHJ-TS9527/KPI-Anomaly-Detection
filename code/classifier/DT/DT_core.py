import sklearn.tree as tree
import numpy as np

class dt():

    def train_DT(self, train_X, train_Y):
        model = tree.DecisionTreeClassifier(random_state=21368)
        model.fit(train_X, train_Y)

        return model

    def test_DT(self, model, test_X):
        pred = model.predict_proba(test_X)
        return np.array(pred[:, 1])
