function feature_extract(KPI_name, type, pyramid, path)
    % This function is used to extract the feature of KPI curves
    % Input arguments:
    % KPI name: the name of KPI [string]
    % type: 0, training data; 1, test data [integer]
    % pyramid: the window size of global feature extraction [row vector]
    % path: the path of file [4-by-1 string cell], each row is:
    % source_train_path, source_test_path, target_train_path,
    % target_test_path
    
    %% Get file path
    if type % test data
        source_data_csv_file = [path{2, 1} KPI_name '.csv'];
        target_data_csv_file = [path{4, 1} KPI_name '.csv'];
    else % train data
        source_data_csv_file = [path{1, 1} KPI_name '.csv'];
        target_data_csv_file = [path{3, 1} KPI_name '.csv'];
    end
    %% Load origin time series data
    origin_data = csvread(source_data_csv_file);
    record_number = size(origin_data, 1);
    if record_number < max(pyramid)
        error('The data size is too small');
    end
    %% Local feature extraction
    start_idx = max(pyramid);
    if start_idx < 4
        error('The max window size is too large');
    end
    timestamps = origin_data(start_idx:end, 1);
    labels = origin_data(start_idx:end, 3);
    peak_values = origin_data(start_idx:end, 2);
    diff1 = origin_data(start_idx:end, 2) - origin_data(start_idx - 1:end - 1, 2);
    diff2 = origin_data(start_idx:end, 2) + origin_data(start_idx - 2:end - 2, 2) - 2 * origin_data(start_idx - 1, 2);
    diff3 = origin_data(start_idx:end, 2) + 3 * origin_data(start_idx - 2:end-2, 2) - 3 * origin_data(start_idx - 1, 2) - ...
        origin_data(start_idx - 3:end - 3, 2);
    monotonicity = diff1 .* (origin_data(start_idx - 1:end - 1, 2) - origin_data(start_idx - 2:end -2, 2));
    concavity = diff2 .* (origin_data(start_idx - 1:end - 1, 2) + origin_data(start_idx - 3:end - 3, 2) - 2 * origin_data(start_idx - 2:end - 2, 2));
    %% Global feature extraction - STL
    if type
        STL_data = csvread([path{4,1} 'STL_' KPI_name '.csv']);
    else
        STL_data = csvread([path{3,1} 'STL_' KPI_name '.csv']);
    end
    %% Window global feature extraction
    EWT_params.N = 5;
    EWT_params.SamplingRate = -1;
    EWT_params.globtrend = 'none';
    EWT_params.reg = 'none';
    EWT_params.detect = 'locmax';
    EWT_params.completion = 1;
    EWT_params.log = 0;
    frame_number = record_number - start_idx + 1;
    window_feature = cell(frame_number, length(pyramid));
    for window_size_idx = 1:length(pyramid)
        window_size = pyramid(window_size_idx);
        lpc_order = min([window_size 13]);
        for timestamp_idx = start_idx:record_number
            frame = origin_data(timestamp_idx - window_size + 1:timestamp_idx, 2).';
            % statistic feature
            mean_value = mean(frame);
            standard_dev = std(frame);
            moments = zeros(1, 3);
            for moment_order = 1:3
                moments(moment_order) = moment(frame, moment_order + 2);
            end
            KURTOSIS = kurtosis(frame);
            SKEWNESS = skewness(frame);
            % wavelet transfor feature
            [wave_c1, level1] = wavedec(frame, 9, 'db2');
            [wave_c2, level2] = wavedec(frame, 9, 'dmey');
            wavedetailcoef1 = detcoef(wave_c1, level1, 1:9);
            wavec1_feature = [wavedetailcoef1{4}(end) wavedetailcoef1{5}(end) wavedetailcoef1{6}(end)...
                wavedetailcoef1{7}(end) wavedetailcoef1{8}(end) wavedetailcoef1{9}(end)];
            wavedetailcoef2 = detcoef(wave_c2, level2, 1:9);
            wavec2_feature = [wavedetailcoef2{4}(end) wavedetailcoef2{5}(end) wavedetailcoef2{6}(end)...
                wavedetailcoef2{7}(end) wavedetailcoef2{8}(end) wavedetailcoef2{9}(end)];
            [ewt, mfb, ~] = EWT1D(frame', EWT_params);
            IMFs = Modes_EWT1D(ewt, mfb);
            IMF_feature = [IMFs{1}(end) IMFs{2}(end) IMFs{3}(end) IMFs{4}(end) IMFs{5}(end)];
            % LPC
            LPCC = lpc(frame, lpc_order);
            predict_value = sum(frame(end - lpc_order + 1:end) .* LPCC(2:end));
            % Fractal feature
            hurst = estimate_hurst_exponent(frame);
            % Store
            frame_feature = [mean_value standard_dev moments KURTOSIS SKEWNESS predict_value hurst ...
                wavec1_feature wavec2_feature IMF_feature];
            window_feature{timestamp_idx - start_idx + 1, window_size_idx} = frame_feature;
        end
    end
    %% Merge all features together
    window_data_type = '';
    window_feature_num = 0;
    for pyramid_idx = 1:length(pyramid)
        window_feature_num = window_feature_num + length(window_feature{1, pyramid_idx});
    end
    for window_feature_idx = 1:window_feature_num
        if window_feature_idx == window_feature_num
            window_data_type = [window_data_type '%.20f\n'];
        else
            window_data_type = [window_data_type '%.20f,'];
        end
    end
    features = zeros(frame_number, 2 + 6 + 3 + window_feature_num);
    features(:, 1) = timestamps;
    features(:, 2) = labels;
    features(:, 3) = peak_values;
    features(:, 4) = diff1;
    features(:, 5) = diff2;
    features(:, 6) = diff3;
    features(:, 7) = monotonicity;
    features(:, 8) = concavity;
    features(:, 9:11) = STL_data;
    local_feature_type = '%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,';
    %% Merge the feature of each timestamp window
    for timestamp_idx = 1:frame_number
        w_feature = [];
        for pyramid_idx = 1:length(pyramid)
            w_feature = cat(2, w_feature, window_feature{timestamp_idx, pyramid_idx});
        end
        features(timestamp_idx, 12:end) = w_feature;
    end
    %% Write to csv file
    fid = fopen(target_data_csv_file, 'w');
    for timestamp_idx = 1:frame_number
        fprintf(fid, ['%d,%d,' local_feature_type window_data_type], features(timestamp_idx, :));
    end
    fclose(fid);
end