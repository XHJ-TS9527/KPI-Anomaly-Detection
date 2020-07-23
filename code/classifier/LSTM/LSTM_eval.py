import sys
sys.path.append('../..')
import evaluation.evaluation_core as eval_core


class main_activity():

    def __init__(self):
        self.train_feature_path = '../../../dataset/training_test_feature/train/'
        self.test_feature_path = '../../../dataset/training_test_feature/test/'
        self.model_path = '../../../model/LSTM/'
        self.KPI_name_file_path = '../../../dataset/KPI_names.txt'

    def main(self):
        # feature type
        self.feature_type = 'local+global+temp'
        self.LSTM_layers = [1, 2, 3, 4, 5]
        self.LSTM_layers.reverse()
        # KPI names
        KPI_name_file = open(self.KPI_name_file_path, 'rt')
        KPI_names = KPI_name_file.readlines()
        KPI_name_file.close()
        # estimate the result and write to file
        evaluation_core = eval_core.main_activity_core(self.model_path, 0.5)
        for layer in self.LSTM_layers:
            ACC, REC, AUC, F1, TP, FN, FP, TN = evaluation_core.evaluation_layer(KPI_names, layer)
            eval_text = []
            for idx in range(len(KPI_names)):
                eval_text.append('%s evaluation result:\n' % KPI_names[idx].replace('\n', ''))
                eval_text.append('Acc:%.5f,Recall:%.5f,AUC:%.7f,F1-score:%.7f\nconfussion matrix:\n' % (
                    ACC[idx], REC[idx], AUC[idx], F1[idx]))
                eval_text.append('class\tPredict Positive\tPredict Negative\n')
                eval_text.append('True Positive\t%d\t%d\n' % (TP[idx], FN[idx]))
                eval_text.append('True Negative\t%d\t%d\n' % (FP[idx], TN[idx]))
            eval_file = open('LSTM_evaluation_%d.txt' % layer, 'wt')
            eval_file.writelines(eval_text)
            eval_file.close()
        print('LSTM Experiment Finished.')


if __name__ == '__main__':
    M = main_activity()
    M.main()
