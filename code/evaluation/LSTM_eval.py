class main_activity():

    def __init__(self):
        self.LSTM_path = '../classifier/LSTM/'
        self.LSTM_eval_file_path = '../classifier/LSTM/LSTM_evaluation_'

    def main(self):
        layers = (1, 2, 3, 4, 5)
        ACC = []
        REC = []
        AUC = []
        F1 = []
        for layer in layers:
            acc, recall, auc, f1 = self.merge_evaluation(layer)
            ACC.append(acc)
            REC.append(recall)
            AUC.append(auc)
            F1.append(f1)
        # merge content
        ACC_content = []
        REC_content = []
        AUC_content = []
        F1_content = []
        for idx in range(len(ACC)):
            ACC_content.append('%s,%s,%s,%s,%s\n' % (ACC[0][idx], ACC[1][idx], ACC[2][idx], ACC[3][idx], ACC[4][idx]))
            REC_content.append('%s,%s,%s,%s,%s\n' % (REC[0][idx], REC[1][idx], REC[2][idx], REC[3][idx], REC[4][idx]))
            AUC_content.append('%s,%s,%s,%s,%s\n' % (AUC[0][idx], AUC[1][idx], AUC[2][idx], AUC[3][idx], AUC[4][idx]))
            F1_content.append('%s,%s,%s,%s,%s\n' % (F1[0][idx], F1[1][idx], F1[2][idx], F1[3][idx], F1[4][idx]))
        # write the content
        acc_file = open(self.LSTM_path + 'Accuracy.csv', 'wt')
        acc_file.writelines(ACC_content)
        acc_file.close()
        rec_file = open(self.LSTM_path + 'Recall.csv', 'wt')
        rec_file.writelines(REC_content)
        rec_file.close()
        auc_file = open(self.LSTM_path + 'AUC.csv', 'wt')
        auc_file.writelines(AUC_content)
        auc_file.close()
        f1_file = open(self.LSTM_path + 'F1_score.csv', 'wt')
        f1_file.writelines(F1_content)
        f1_file.close()
        print('Evaluation Merge Finished')

    def merge_evaluation(self, layer):
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
