import sklearn.svm as svm


class SVM():

    def train_SVM(self, train_X, train_Y):
        model = svm.SVC(kernel='rbf', probability=True, random_state=21368)
        model.fit(train_X, train_Y)
        return model

    def test_SVM(self, model, test_X):
        pred = model.predict_proba(test_X)[:, 1]
        return pred.reshape(-1, 1)
