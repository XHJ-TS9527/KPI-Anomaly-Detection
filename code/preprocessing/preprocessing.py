import os
import random
import bisect
import numpy as np
import scipy.interpolate as interp


class main_activity():

    def __init__(self):
        self.origin_path = '../../dataset/select KPIs/'
        self.os_origin_path = os.path.abspath('..') + os.path.abspath('..') + r'\dataset\select KPIs'
        self.target_path = '../../dataset/preprocessed_data/'

    def main(self):
        # get all KPI files
        file_names = []
        for file in os.listdir(self.os_origin_path):
            file_names.append(file)
        KPI_names = []
        # start preprocessing
        for each_file in file_names:
            KPI_names.append('%s\n' % self.single_KPI_processing(each_file))
        # write KPI name file
        file = open('../../dataset/KPI_names.txt', 'wt')
        file.writelines(KPI_names)
        file.close()

    def single_KPI_processing(self, KPI_file_name):
        # get KPI data
        KPI_name = KPI_file_name.split('.')[0]
        print('processing %s' % KPI_name)
        KPI_file = open(self.origin_path + KPI_file_name, 'rt')
        KPI_data = KPI_file.readlines()
        KPI_file.close()
        # reshape the data into numpy form
        # get all should appear timestamps in dataset
        start_record_timestamp = int(KPI_data[0].split(',')[0])
        end_record_timestamp = int(KPI_data[-1].split(',')[0])
        timestamps = [x for x in range(start_record_timestamp, end_record_timestamp, 60)]
        # get all dismissed timestamps
        record_idx = 0
        exist_timestamp = []
        missing_timestamp = []
        exist_values = []
        values = []
        labels = []
        for timestamp in timestamps:
            now_record = KPI_data[record_idx].split(',')
            now_record_timestamp = int(now_record[0])
            if now_record_timestamp == timestamp:
                now_record_value = float(now_record[1])
                now_record_label = int(now_record[2])
                exist_timestamp.append(timestamp)
                exist_values.append(now_record_value)
                values.append(now_record_value)
                labels.append(now_record_label)
                record_idx += 1
            else:
                missing_timestamp.append(timestamp)
                values.append(None)
                labels.append(None)
        # Interp1
        interp_fun = interp.interp1d(np.array(exist_timestamp), np.array(exist_values), kind='cubic')
        missing_values = list(interp_fun(np.array(missing_timestamp)))
        for each_missing_timestamp_idx in range(len(missing_timestamp)):
            # Put the values into the new thing
            each_missing_timestamp = missing_timestamp[each_missing_timestamp_idx]
            missing_idx = timestamps.index(each_missing_timestamp)
            values[missing_idx] = missing_values[each_missing_timestamp_idx]
            # find the nearest exist timestamp
            next_idx = bisect.bisect_left(exist_timestamp, each_missing_timestamp)
            front_idx = next_idx - 1
            front_exist_timestamp = exist_timestamp[front_idx]
            next_exist_timestamp = exist_timestamp[next_idx]
            # infer the label
            to_front_dist = abs(front_exist_timestamp - each_missing_timestamp)
            to_next_dist = abs(next_exist_timestamp - each_missing_timestamp)
            if to_front_dist < to_next_dist:
                labels[missing_idx] = labels[front_idx]
            elif to_next_dist < to_front_dist:
                labels[missing_idx] = labels[next_idx]
            else:
                if labels[front_idx] == labels[next_idx]:
                    labels[missing_idx] = labels[front_idx]
                else:
                    labels[missing_idx] = random.choice([0, 1])
        # write the washed KPI data to file
        KPI_file_content = []
        for idx in range(len(timestamps)):
            KPI_file_content.append('%d,%s,%d\n' % (timestamps[idx], str(values[idx]), labels[idx]))
        KPI_file = open(self.target_path + KPI_file_name, 'wt')
        KPI_file.writelines(KPI_file_content)
        KPI_file.close()
        return KPI_name


if __name__ == '__main__':
    M = main_activity()
    M.main()
