%% Initialization
close all;
clear;
clc;
format long;
%% Parameter setting
pyramid_window_size = [10 30 60 1440];
%% Load KPI names
KPI_names = importdata('../../dataset/KPI_names.txt');
%% File I/O Path
source_train_path = '../../dataset/training_test_data/train/';
source_test_path = '../../dataset/training_test_data/test/';
target_train_path = '../../dataset/training_test_feature/train/';
target_test_path = '../../dataset/training_test_feature/test/';
path = {source_train_path; source_test_path; target_train_path; target_test_path};
%% Processing
try
    parfor KPI_idx = 1:size(KPI_names, 1)
        disp(['Processing ' KPI_names{KPI_idx, 1} ' test data feature extraction']);
        feature_extract(KPI_names{KPI_idx, 1}, 1, pyramid_window_size, path); % extract the feature of testing data
        disp(['Processing ' KPI_names{KPI_idx, 1} ' train data feature extraction']);
        feature_extract(KPI_names{KPI_idx, 1}, 0, pyramid_window_size, path); % extract the feature of training data
    end
    disp('Feature Extraction Success');
catch
end
quit;