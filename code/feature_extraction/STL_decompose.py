import numpy as np
import statsmodels.api as sm

class main_acitivity():

    def __init__(self):
        self.origin_train_path = '../../dataset/training_test_data/train/'
        self.origin_test_path = '../../dataset/training_test_data/test/'
        self.target_train_path = '../../dataset/training_test_feature/train/'
        self.target_test_path = '../../dataset/training_test_feature/test/'
        self.start_idx = 1439

    def main(self):
        KPI_name_file = open('../../dataset/KPI_freqs.csv', 'rt')
        KPI_freqs = KPI_name_file.readlines()
        KPI_name_file.close()
        for each_KPI in KPI_freqs:
            KPI_detail = each_KPI.replace('\n', '').split(',')
            print('processing %s training data' % KPI_detail[0])
            self.decompose(KPI_detail[0], int(KPI_detail[1]), 0) # training data STL
            print('processing %s testing data' % KPI_detail[0])
            self.decompose(KPI_detail[0], int(KPI_detail[2]), 1)  # test data STL

    def decompose(self, KPI_name, freq, type):
        if type: # test data
            KPI_file = open(self.origin_test_path + '%s.csv' % KPI_name, 'rt')
        else: # train data
            KPI_file = open(self.origin_train_path + '%s.csv' % KPI_name, 'rt')
        # get data
        KPI_data = KPI_file.readlines()
        KPI_file.close()
        KPI_values = []
        for each in KPI_data:
            KPI_values.append(float(each.split(',')[1]))
        KPI_values = KPI_values[self.start_idx:]
        # STL decompose
        STL = self.STL_decomposition(KPI_values, freq)
        season = np.array(list(STL.seasonal))
        trend = np.array(list(STL.trend))
        res = np.array(list(STL.resid))
        trend = self.replace_nan(trend)
        res = self.replace_nan(res)
        # write file
        write_content = []
        for idx in range(len(KPI_values)):
            write_content.append('%.16f,%.16f,%.16f\n' % (season[idx], trend[idx], res[idx]))
        if type:
            target_file = open(self.target_test_path + 'STL_%s.csv' % KPI_name, 'wt')
        else:
            target_file = open(self.target_train_path + 'STL_%s.csv' % KPI_name, 'wt')
        target_file.writelines(write_content)
        target_file.close()

    def STL_decomposition(self,time_series, frequency):
        return sm.tsa.seasonal_decompose(np.array(time_series), freq=frequency)

    def replace_nan(self, array):
        start_nan_idx = 0
        end_nan_idx = -1
        while np.isnan(array[start_nan_idx]):
            start_nan_idx += 1
        while np.isnan(array[end_nan_idx]):
            end_nan_idx -= 1
        array[:start_nan_idx] = array[start_nan_idx]
        array[end_nan_idx:] = array[end_nan_idx]
        return array


if __name__ == '__main__':
    M = main_acitivity()
    M.main()
