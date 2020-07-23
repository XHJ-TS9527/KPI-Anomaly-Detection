%% Initialization
close all;
clear;
clc;
%% KPI names and paths
KPI_names = importdata('../../dataset/KPI_names.txt');
KPI_source_train_path = '../../dataset/training_test_data/train/';
KPI_source_test_path = '../../dataset/training_test_data/test/';
start_idx = 1440;
%% Calculate the frequency of each KPI using autocorrelation
KPI_freq = zeros(size(KPI_names, 1), 2); %column1: training data, column2: testing data
for KPI_idx = 1:size(KPI_names, 1)
    KPI_name = KPI_names{KPI_idx, 1};
    % Training data
    KPI_data = csvread([KPI_source_train_path KPI_name '.csv']);
    KPI_freq(KPI_idx, 1) = get_freq(KPI_data(:, 2), start_idx);
    % Testing data
    KPI_data = csvread([KPI_source_test_path KPI_name '.csv']);
    KPI_freq(KPI_idx, 2) = get_freq(KPI_data(:, 2), start_idx);
end
%% Store the frequency
fid = fopen('../../dataset/KPI_freqs.csv', 'wt');
for KPI_idx = 1:size(KPI_names, 1)
    KPI_name = KPI_names{KPI_idx, 1};
    fprintf(fid, '%s,%d,%d\n', KPI_name, KPI_freq(KPI_idx, 1), KPI_freq(KPI_idx, 2));
end
fclose(fid);
%% Peroid estimation function
function frequency = get_freq(signal, start_idx)
    % This function is used to get the periodc of a signal using
    % autocorrelation method.
    % Input arguments:
    % signal: the signal to estimate periodc [column vector]
    % start_idx: the start index to estimate
    % Output argument:
    % frequency: the estimate frequency [positive integer]
    
    corr = fftshift(xcorr(signal(start_idx:end).'));
    start_rising_idx = 1;
    while corr(start_rising_idx + 1) < corr(start_rising_idx)
        if start_rising_idx == length(corr)
            break;
        else
            start_rising_idx = start_rising_idx + 1;
        end
    end
    end_falling_idx = length(corr);
    while corr(end_falling_idx - 1) <  corr(end_falling_idx)
        if end_falling_idx == 1
            break;
        else
            end_falling_idx = end_falling_idx - 1;
        end
    end
    if start_rising_idx == length(corr) || end_falling_idx == 1 || start_rising_idx == end_falling_idx
        sim_idx = [];
    else
        max_corr = max(corr(start_rising_idx:end_falling_idx));
        threshold = 0.2 * max_corr;
        corr = corr(1:length(signal(start_idx:end)));
        sim_idx = find(corr > threshold);
        sim_idx(sim_idx < start_rising_idx) = [];
        sim_idx(sim_idx > end_falling_idx) = [];
    end
    if isempty(sim_idx)
        freq = length(signal(start_idx:end));
    else
        freq = sim_idx(1);
        while corr(freq + 1) > corr(freq)
            freq = freq + 1;
        end
    end
    frequency = min([freq floor(length(signal(start_idx:end)) / 2)]);
end