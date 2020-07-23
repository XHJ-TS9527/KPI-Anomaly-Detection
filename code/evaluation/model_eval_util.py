import numpy as np
import sklearn.metrics as metrics


class model_eval_util():

    def get_ROC(self,groundtruth, prediction):
        truth = np.array(groundtruth)
        pred = np.array(prediction)
        FPR, TPR, _ = metrics.roc_curve(truth, pred)
        return FPR, TPR

    def get_AUC(self, groundtruth, prediction):
        truth = np.array(groundtruth)
        pred = np.array(prediction)
        return metrics.roc_auc_score(truth, pred)

    def get_PRC(self, groundtruth, prediction):
        truth = np.array(groundtruth)
        pred = np.array(prediction)
        PRC, REC, _ = metrics.precision_recall_curve(truth, pred)
        return PRC, REC

    def get_F1_score(self, groundtruth, prediction_class):
        truth = np.array(groundtruth)
        pred = np.array(prediction_class)
        TN, FP, FN, TP = self.get_confusion_mtx(truth, pred)
        try:
            recall = TP / (TP + FN)
        except:
            recall = 0
        try:
            precision = TP / (TP + FP)
        except:
            precision = 0
        f1_score = 2 * recall * precision / (precision + recall)
        return f1_score

    def get_accuracy(self, groundtruth, prediction_class):
        truth = np.array(groundtruth)
        pred = np.array(prediction_class)
        return metrics.accuracy_score(truth, pred)

    def get_recall(self, groundtruth, prediction_class):
        truth = np.array(groundtruth)
        pred = np.array(prediction_class)
        TN, FP, FN, TP = self.get_confusion_mtx(groundtruth, pred)
        recall = TP / (TP + FN)
        return recall

    def get_confusion_mtx(self, groundtruth, prediction_class):
        truth = np.array(groundtruth)
        pred = np.array(prediction_class)
        try:
            TN, FP, FN, TP = metrics.confusion_matrix(truth, pred).ravel()
        except:
            TN = len(truth)
            FP = 0
            FN = 0
            TP = 0
        return TN, FP, FN, TP

    def prob2cls(self, prediction, threshold):
        prediction[prediction < threshold] = 0
        prediction[prediction >= threshold] = 1
        return prediction
