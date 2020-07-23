import numpy as np


class NBNN():

    def NBNN_classify(self, train_X, train_Y, test_X):
        negative_sample = train_X[train_Y == 0, :]
        positive_sample = train_X[train_Y == 1, :]
        pred = np.zeros((test_X.shape[0], 1), dtype=np.int8)
        for sample_idx in range(test_X.shape[0]):
            to_negative_dist = np.abs(test_X[sample_idx, :] - negative_sample)
            to_positive_dist = np.abs(test_X[sample_idx, :] - positive_sample)
            to_negative_min = np.amin(to_negative_dist, axis=0)
            to_positive_min = np.amin(to_positive_dist, axis=0)
            if np.sum(to_negative_min) < np.sum(to_positive_min):
                pred[sample_idx, 0] = 0
            else:
                pred[sample_idx, 0] = 1
        return pred.reshape(-1, 1)
