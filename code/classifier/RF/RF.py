import sys
sys.path.append('../..')
import RF_core as core
import numpy as np
import pandas as pd
import pickle as plk
import concurrent.futures
import evaluation.evaluation_core as eval_core

class main_activity():

    def __init__(self):
        self.train_feature_path = '../../../dataset/training_test_feature/train/'
        self.test_feature_path = '../../../dataset/training_test_feature/test/'
        self.model_path = '../../../model/RF/'
        self.KPI_name_file_path = '../../../dataset/KPI_names.txt'

    def main(self):
        # feature type
        self.feature_type = 'local+global+temp'
        # KPI names
        KPI_name_file = open(self.KPI_name_file_path, 'rt')
        KPI_names = KPI_name_file.readlines()
        KPI_name_file.close()
        with concurrent.futures.ProcessPoolExecutor() as executor:  # multi CPUs processing
            executor.map(self.real_RF, KPI_names)
        '''
        for each_KPI in KPI_names: # one CPU processing
            self.real_RF(each_KPI)
        '''
        # estimate the result and write to file
        evaluation_core = eval_core.main_activity_core(self.model_path, 0.5)
        ACC, REC, AUC, F1, TP, FN, FP, TN = evaluation_core.evaluation(KPI_names)
        eval_text = []
        for idx in range(len(KPI_names)):
            eval_text.append('%s evaluation result:\n' % KPI_names[idx].replace('\n', ''))
            eval_text.append('Acc:%.5f,Recall:%.5f,AUC:%.7f,F1-score:%.7f\nconfussion matrix:\n' % (
                ACC[idx], REC[idx], AUC[idx], F1[idx]))
            eval_text.append('class\tPredict Positive\tPredict Negative\n')
            eval_text.append('True Positive\t%d\t%d\n' % (TP[idx], FN[idx]))
            eval_text.append('True Negative\t%d\t%d\n' % (FP[idx], TN[idx]))
        eval_file = open('RF_evaluation.txt', 'wt')
        eval_file.writelines(eval_text)
        eval_file.close()
        print('RF Experiment Finished.')

    def real_RF(self, each_KPI):
        KPI_name = each_KPI.replace('\n', '')
        # Feature selection
        print('========Processing %s feature selection===================' % KPI_name)
        train_data, train_label, test_data, test_label = self.feature_selection(KPI_name, self.feature_type)
        print('========Processing %s RF model training==============' % KPI_name)
        # Train RF model
        rfc = core.rf()
        model = rfc.train_RF(train_data, train_label)
        # Evaluate the NB model
        print('========Processing %s RF model testing===============' % KPI_name)
        pred = rfc.test_RF(model, test_data)
        # Save the model
        plk.dump(model, open(self.model_path + '%s.model' % KPI_name, 'wb'))
        # write the prediction result, if need ROC curve
        testing_text = []
        for idx in range(len(pred)):
            testing_text.append('%d,%.10f\n' % (test_label[idx], pred[idx]))
        testing_pred_file = open(self.model_path + '%s_test.csv' % KPI_name, 'wt')
        testing_pred_file.writelines(testing_text)
        testing_pred_file.close()

    def feature_selection(self, KPI_name, feature_type):
        # Load training dataset
        training_dataset = pd.read_csv(self.train_feature_path + '%s.csv' % KPI_name, low_memory=False).values.astype(
            float)
        testing_dataset = pd.read_csv(self.test_feature_path + '%s.csv' % KPI_name, low_memory=False).values.astype(
            float)
        # Feature selection
        train_data_size = training_dataset.shape[0]
        test_data_size = testing_dataset.shape[0]
        training_data = np.empty(shape=[train_data_size, 0])
        testing_data = np.empty(shape=[test_data_size, 0])
        training_label = np.array(training_dataset[:, 1])
        testing_label = np.array(testing_dataset[:, 1])
        if 'local' in feature_type:
            training_data = np.concatenate((training_data, training_dataset[:, 2:8]), axis=1)
            testing_data = np.concatenate((testing_data, testing_dataset[:, 2:8]), axis=1)
        if 'global' in feature_type:
            training_data = np.concatenate((training_data, training_dataset[:, 8:11]), axis=1)
            testing_data = np.concatenate((testing_data, testing_dataset[:, 8:11]), axis=1)
        if 'temp' in feature_type:
            training_data = np.concatenate((training_data, training_dataset[:, 11:]), axis=1)
            testing_data = np.concatenate((testing_data, testing_dataset[:, 11:]), axis=1)
        # Mapminmax
        training_data = self.mapminmax(training_data)
        testing_data = self.mapminmax(testing_data)
        return training_data, training_label, testing_data, testing_label

    def mapminmax(self, dataset):
        for col_idx in range(dataset.shape[1]):
            dataset[:, col_idx] = np.nan_to_num(dataset[:, col_idx], nan=0.0, posinf=np.inf, neginf=-np.inf)
            without_inf = np.delete(dataset[:, col_idx], np.where(np.isinf(dataset[:, col_idx])))
            pos_inf_idx = np.where(np.isposinf(dataset[:, col_idx]))
            neg_inf_idx = np.where(np.isneginf(dataset[:, col_idx]))
            max_value = np.max(without_inf)
            min_value = np.min(without_inf)
            if max_value == min_value:
                dataset[:, col_idx] = 0.5
            else:
                dataset[pos_inf_idx, col_idx] = max_value
                dataset[neg_inf_idx, col_idx] = min_value
                dataset[:, col_idx] = (dataset[:, col_idx] - min_value) / (max_value - min_value)
        return dataset


if __name__ == '__main__':
    M = main_activity()
    M.main()
