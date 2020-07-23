import sys
sys.path.append('..')
import evaluation.model_eval_util as util
import pandas as pd
import numpy as np


class main_activity_core():

    def __init__(self, model_path, threshold):
        self.model_path = model_path
        self.threshold = threshold

    def evaluation(self, KPI_names):
        # open the namelist file
        ACC = []
        REC = []
        AUC = []
        F1 = []
        TP = []
        FN = []
        FP = []
        TN = []
        self.eval_util = util.model_eval_util()
        for each_KPI in KPI_names:
            KPI_name = each_KPI.replace('\n', '')
            print('------>Processing %s Evaluation---' % KPI_name)
            acc, recall, auc, F1_score, tp, fn, fp, tn, fpr, tpr, prc, rec = self.real_eval(KPI_name)
            ACC.append(acc)
            REC.append(recall)
            AUC.append(auc)
            F1.append(F1_score)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            TN.append(tn)
            # write ROC curve and PRC curve to file
            roc_text = []
            prc_text = []
            for idx in range(len(fpr)):
                roc_text.append('%.10f,%.10f\n' % (fpr[idx], tpr[idx]))
            for idx in range(len(prc)):
                prc_text.append('%.10f,%.10f\n' % (rec[idx], prc[idx]))
            ROC_file = open(self.model_path + 'ROC_%s.csv' % KPI_name, 'wt')
            ROC_file.writelines(roc_text)
            ROC_file.close()
            PRC_file = open(self.model_path + 'PRC_%s.csv' % KPI_name, 'wt')
            PRC_file.writelines(prc_text)
            PRC_file.close()
        return ACC, REC, AUC, F1, TP, FN, FP, TN

    def evaluation_layer(self, KPI_names, layer):
        # open the namelist file
        ACC = []
        REC = []
        AUC = []
        F1 = []
        TP = []
        FN = []
        FP = []
        TN = []
        self.eval_util = util.model_eval_util()
        for each_KPI in KPI_names:
            KPI_name = each_KPI.replace('\n', '')
            print('------>Processing %s Evaluation with layer number %d---' % (KPI_name, layer))
            acc, recall, auc, F1_score, tp, fn, fp, tn, fpr, tpr, prc, rec = self.real_eval_layer(KPI_name, layer)
            ACC.append(acc)
            REC.append(recall)
            AUC.append(auc)
            F1.append(F1_score)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            TN.append(tn)
            # write ROC curve and PRC curve to file
            roc_text = []
            prc_text = []
            for idx in range(len(fpr)):
                roc_text.append('%.10f,%.10f\n' % (fpr[idx], tpr[idx]))
            for idx in range(len(prc)):
                prc_text.append('%.10f,%.10f\n' % (rec[idx], prc[idx]))
            ROC_file = open(self.model_path + 'ROC_%s_%d.csv' % (KPI_name, layer), 'wt')
            ROC_file.writelines(roc_text)
            ROC_file.close()
            PRC_file = open(self.model_path + 'PRC_%s_%d.csv' % (KPI_name, layer), 'wt')
            PRC_file.writelines(prc_text)
            PRC_file.close()
        return ACC, REC, AUC, F1, TP, FN, FP, TN

    def real_eval(self, KPI_name):
        # load predict result
        predict_result = pd.read_csv(self.model_path + '%s_test.csv' % KPI_name).values.astype(float)
        #change to class
        pred_class = self.eval_util.prob2cls(predict_result[:, 1], self.threshold)
        # evaluate
        if len(np.unique(predict_result[:, 0])) > 1:
            FPR, TPR = self.eval_util.get_ROC(predict_result[:, 0], predict_result[:, 1])
            AUC = self.eval_util.get_AUC(predict_result[:, 0], predict_result[:, 1])
            PRC, REC = self.eval_util.get_PRC(predict_result[:, 0], predict_result[:, 1])
        else:
            FPR = []
            TPR = []
            PRC = []
            REC = []
            AUC = -1
        F1_score = self.eval_util.get_F1_score(predict_result[:, 0], pred_class)
        acc = self.eval_util.get_accuracy(predict_result[:, 0], pred_class)
        recall = self.eval_util.get_recall(predict_result[:, 0], pred_class)
        TN, FP, FN, TP = self.eval_util.get_confusion_mtx(predict_result[:, 0], pred_class)
        return acc, recall, AUC, F1_score, TP, FN, FP, TN, FPR, TPR, PRC, REC

    def real_eval_layer(self, KPI_name, layer):
        # load predict result
        predict_result = pd.read_csv(self.model_path + '%s_%d_test.csv' % (KPI_name, layer)).values.astype(float)
        #change to class
        pred_class = self.eval_util.prob2cls(predict_result[:, 1], self.threshold)
        # evaluate
        if len(np.unique(predict_result[:, 0])) > 1:
            FPR, TPR = self.eval_util.get_ROC(predict_result[:, 0], predict_result[:, 1])
            AUC = self.eval_util.get_AUC(predict_result[:, 0], predict_result[:, 1])
            PRC, REC = self.eval_util.get_PRC(predict_result[:, 0], predict_result[:, 1])
        else:
            FPR = []
            TPR = []
            PRC = []
            REC = []
            AUC = -1
        F1_score = self.eval_util.get_F1_score(predict_result[:, 0], pred_class)
        acc = self.eval_util.get_accuracy(predict_result[:, 0], pred_class)
        recall = self.eval_util.get_recall(predict_result[:, 0], pred_class)
        TN, FP, FN, TP = self.eval_util.get_confusion_mtx(predict_result[:, 0], pred_class)
        return acc, recall, AUC, F1_score, TP, FN, FP, TN, FPR, TPR, PRC, REC
