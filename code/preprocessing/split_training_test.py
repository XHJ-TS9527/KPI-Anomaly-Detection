class main_activity():
    def __init__(self):
        self.origin_path = '../../dataset/preprocessed_data/'
        self.target_train_path = '../../dataset/training_test_data/train/'
        self.target_test_path = '../../dataset/training_test_data/test/'

    def main(self):
        KPI_names_file = open('../../dataset/KPI_names.txt', 'rt')
        KPI_names = KPI_names_file.readlines()
        KPI_names_file.close()
        for each_KPI in KPI_names:
            self.split_dataset(each_KPI.replace('\n', ''))

    def split_dataset(self, KPI_name):
        origin_data_file = open(self.origin_path + '%s.csv' % KPI_name, 'rt')
        KPI_data = origin_data_file.readlines()
        origin_data_file.close()
        cut_idx = int(0.8 * len(KPI_data))
        train_data = KPI_data[0: cut_idx + 1]
        test_data = KPI_data[cut_idx + 1:]
        training_data_file = open(self.target_train_path + '%s.csv' % KPI_name, 'wt')
        training_data_file.writelines(train_data)
        training_data_file.close()
        testing_data_file = open(self.target_test_path + '%s.csv' % KPI_name, 'wt')
        testing_data_file.writelines(test_data)
        testing_data_file.close()


if __name__ == '__main__':
    M = main_activity()
    M.main()
