import numpy as np


class main_activity():

    def __init__(self):
        self.model_path = '../classifier/'
        self.LSTM_eval_file_path = '../classifier/LSTM/LSTM_evaluation_'

    def main(self):
        # test detectors
        models = ('LSTM', 'ANN', 'BLS', 'OC_SVM', 'SVM', 'NB', 'NBNN', 'DT', 'RF', 'XGBoost')
        # extract the test result of the models
        ACC = []
        RECALL = []
        AUC = []
        F1 = []
        for detector in models:
            acc, recall, auc, f1 = self.merge_evaluation(detector)
            ACC.append(acc)
            RECALL.append(recall)
            AUC.append(auc)
            F1.append(f1)
        # write the contents
        acc_file = open('Accuracy.csv', 'wt')
        acc_file.writelines(ACC)
        acc_file.close()
        rec_file = open('Recall.csv', 'wt')
        rec_file.writelines(RECALL)
        rec_file.close()
        auc_file = open('AUC.csv', 'wt')
        auc_file.writelines(AUC)
        auc_file.close()
        f1_file = open('F1_score.csv', 'wt')
        f1_file.writelines(F1)
        f1_file.close()
        print('Evaluation Merge Finished')

    def merge_evaluation(self, detector):
        ACC = []
        REC = []
        AUC = []
        F1 = []
        if detector == 'LSTM':
            for layer in range(1,6):
                acc, rec, auc, f1 = self.merge_evaluation_layer(layer)
                ACC.append(acc)
                REC.append(rec)
                AUC.append(auc)
                F1.append(f1)
            ACC = np.array(ACC, dtype=np.float)
            REC = np.array(REC, dtype=np.float)
            AUC = np.array(AUC, dtype=np.float)
            F1 = np.array(F1, dtype=np.float)
            F1 = np.nan_to_num(F1)
            ACC = list(np.max(ACC, axis=0))
            REC = list(np.max(REC, axis=0))
            AUC = list(np.max(AUC, axis=0))
            F1 = list(np.max(F1, axis=0))
            for idx in range(len(ACC)):
                ACC[idx] = '%.7f' % ACC[idx]
                REC[idx] = '%.7f' % REC[idx]
                AUC[idx] = '%.7f' % AUC[idx]
                F1[idx] = '%.7f' % F1[idx]
        else:
            if detector == 'SVM':
                final_detector = 'SVM_PCA'
            else:
                final_detector = detector
            # open evaluation file
            eval_file = open('%s%s/%s_evaluation.txt' % (self.model_path, detector, final_detector), 'rt')
            eval_info = eval_file.readlines()
            eval_file.close()
            # Merge the eval info
            idx = 0
            flag = 0
            while idx < len(eval_info):
                if flag:
                    flag = 0
                    content = eval_info[idx].replace('\n', '').split(',')
                    ACC.append(content[0].split(':')[-1])
                    REC.append(content[1].split(':')[-1])
                    AUC.append(content[2].split(':')[-1])
                    F1.append(content[3].split(':')[-1])
                else:
                    if 'result' in eval_info[idx]:
                        flag = 1
                idx += 1
        return '%s\n' % ','.join(ACC), '%s\n' % ','.join(REC),  '%s\n' % ','.join(AUC), '%s\n' % ','.join(F1)

    def merge_evaluation_layer(self, layer):
        # open evaluation file
        eval_file = open(self.LSTM_eval_file_path + str(layer) + '.txt', 'rt')
        eval_info = eval_file.readlines()
        eval_file.close()
        # Merge the eval info
        ACC = []
        REC = []
        AUC = []
        F1 = []
        idx = 0
        flag = 0
        while idx < len(eval_info):
            if flag:
                flag = 0
                content = eval_info[idx].replace('\n', '').split(',')
                ACC.append(content[0].split(':')[-1])
                REC.append(content[1].split(':')[-1])
                AUC.append(content[2].split(':')[-1])
                F1.append(content[3].split(':')[-1])
            else:
                if 'result' in eval_info[idx]:
                    flag = 1
            idx += 1
        return ACC, REC, AUC, F1


if __name__ == '__main__':
    M = main_activity()
    M.main()
