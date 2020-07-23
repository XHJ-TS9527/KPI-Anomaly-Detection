from sklearn.ensemble import RandomForestClassifier
import numpy as np

class rf():

    def __init__(self):
        self.n_estimators_options = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        self.best_n_estimators = 0
        self.best_acc = 0

    def train_RF(self, train_X, train_Y):
        for n_estimators_size in self.n_estimators_options:
            model = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators_size)
            model.fit(train_X, train_Y)
            predict = model.predict(train_X)
            acc = (train_Y == predict).mean()
            # 更新最优参数和 acc
            if acc >= self.best_acc:
                self.best_acc = acc
                self.best_n_estimators = n_estimators_size
        model = RandomForestClassifier(n_jobs=-1, n_estimators=self.best_n_estimators)
        model.fit(train_X, train_Y)
        return model

    def test_RF(self, model, test_X):
        pred = model.predict_proba(test_X)
        return np.array(pred[:, 1])
