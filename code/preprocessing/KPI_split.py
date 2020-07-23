class main_activity():

    def __init__(self):
        self.dataset_path = '../../dataset/origin_data/'

    def main(self):
        KPI_names = []
        # split KPI
        for idx in range(2):
            KPI_names.append(self.KPI_seperation(idx + 1))
        all_have_KPI = KPI_names[0] & KPI_names[1]
        print('共有KPI:')
        print(all_have_KPI)

    def KPI_seperation(self, idx):
        # load dataset file
        dataset_csv = open(self.dataset_path + 'dataset%d.csv' % idx, 'rt')
        KPI_data = dataset_csv.readlines()
        dataset_csv.close()
        del KPI_data[0]
        # KPI dict
        KPI_dict = dict()
        KPI_names = set()
        # Segmentation
        for each_KPI_record in KPI_data:
            KPI_record = each_KPI_record.replace('\n', '').split(',')
            KPI_name = KPI_record[-1]
            this_KPI_records = KPI_dict.get(KPI_name, [])
            this_KPI_records.append('%s\n' % ','.join(KPI_record[0:3]))
            KPI_dict[KPI_name] = this_KPI_records
            KPI_names.add(KPI_name)
        # write to CSV file
        CSV_store_path = '../../dataset/dataset%d_split/' % idx
        for each_KPI in KPI_dict:
            KPI_data = KPI_dict[each_KPI]
            file = open(CSV_store_path + '%s.csv' % each_KPI, 'wt')
            file.writelines(KPI_data)
            file.close()
        return KPI_names


if __name__ == '__main__':
    M = main_activity()
    M.main()
