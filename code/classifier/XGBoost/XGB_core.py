from xgboost.sklearn import XGBClassifier
import numpy as np


class XGB():

    def train_XGB(self, training_X, training_Y):
        """
        :param training_X: train X numpy array
        :param training_Y: train Y numpy array
        :return: XGB model
        """
        xgb_classifier = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.09,
            scale_pos_weight=1,
            seed=21368)
        # map Y to -1, +1
        training_Y[training_Y == 0] = -1
        xgb_classifier.fit(training_X, training_Y)
        return xgb_classifier

    def test_XGB(self, model, testing_X):
        """
        :param model: XGB model
        :param testing_X: test X numpy array
        :return: prediction probability
        """
        return np.array(model.predict_proba(testing_X)[:, 1])
