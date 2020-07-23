import sklearn.svm as svm


class oc_svm():

    def train_oc_svm(self, train_X):
        model = svm.OneClassSVM(gamma='auto')
        model.fit(train_X)
        return model

    def test_oc_svm(self, model, test_X):
        pred = model.predict(test_X)
        return pred.reshape(-1, 1)
