import mne
import numpy as np
import pandas as pd
from mne_features.feature_extraction import extract_features

EEG_COLS = ['Raw Channel Z', 'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four']
MOVEMENT_COL = 'Movement'
SEPARATOR = ';'
FS = 500
LOW_CUTOFF = 0.01
HIGH_CUTOFF = 5
WINDOW_SIZE = 0.05  # second windows
OVERLAP = 0 # 50 ms 

# Load and preprocess the data
def preprocess_data(raw_csv_file, eeg_cols, movement_col, separator, sampling_rate, low_cutoff, high_cutoff):
    eeg_data_df = pd.read_csv(raw_csv_file, sep=separator, usecols=eeg_cols)
    movement_data_df = pd.read_csv(raw_csv_file, sep=separator, usecols=[movement_col])
    movement_data_df['Movement'] = movement_data_df['Movement'].replace(2, 1) # change 2 to 1
    movement_data_df['Movement'] = movement_data_df['Movement'].replace(3, 0) # ignore movement = 3
    movement_data_df['Movement'] = movement_data_df['Movement'].replace(4, 0) # ignore movement = 4

    mne_info = mne.create_info(ch_names=eeg_cols + [movement_col], sfreq=sampling_rate)
    eeg_raw = mne.io.RawArray(np.vstack([eeg_data_df.T.to_numpy() * 1e-6, movement_data_df.T.to_numpy()]), info=mne_info)

    # IIR butterworth filter for EEG channels only
    bandpass_filter = eeg_raw.copy().filter(
        low_cutoff, high_cutoff, picks=eeg_cols, method='iir')
    
    return bandpass_filter

def calculate_corr_coeff(csv_file, matrix_file):
    # Load EEG data from CSV file
    df_features = pd.read_csv(csv_file, sep=",", header=0)

    # Convert columns to numeric data types
    df_features = df_features.apply(pd.to_numeric)

    # Calculate the mean for every 5 columns corresponding to each feature
    feature_mean = pd.DataFrame()
    feature_names = ['kurtosis', 'std', 'min', 'slope']
    for i, feature_name in enumerate(feature_names):
        feature_mean[feature_name] = df_features.iloc[:, i*5:(i+1)*5].mean(axis=1)

    # Calculate the correlation coefficient between mean values of feature pairs
    corr_matrix = feature_mean.corr(method='pearson')
    corr_matrix.to_csv(matrix_file, index=False)
    print(corr_matrix)
    print(f'Correlation matrix saved in {matrix_file}')

# Segment the data into 2-second windows with overlap
def segment_data(data, window_size, events, event_id):
    epochs = mne.Epochs(data, events, event_id, tmin=-window_size, tmax=0, proj=True, baseline=None, preload=True)
    
    return epochs

def mne_extract_features(segmented_data):
    data = segmented_data.get_data()

    mrcp_features = {
        #'variance': 'var',
        'kurtosis': 'kurt',
        #'mean': 'mean',
        'std': 'std'
    }

    all_except_last_channel = data[:, :-1, :]
    last_channel = data[:, -1:, :]
    last_channel_reshaped = last_channel.reshape(last_channel.shape[0], -1)
    last_channel_csv = pd.DataFrame(last_channel_reshaped)
    last_channel_csv.to_csv('Filtering\data\extracted_features\last_channel.csv', index=False)
    print(f'Features saved in last_channel.csv')

    X_mne = extract_features(all_except_last_channel, segmented_data.info['sfreq'], mrcp_features)

    # Custom features
    X_min = np.apply_along_axis(min_func, -1, all_except_last_channel)
    X_slope = np.apply_along_axis(slope_func, -1, all_except_last_channel)
    #X_median = np.apply_along_axis(median_func, -1, all_except_last_channel)

    # Concatenate custom features with MNE features
    X_new = np.hstack((X_min, X_slope, X_mne))

    X_mov = np.apply_along_axis(max_func, -1, last_channel[:, :, -25:]) 
    unique_counts, unique_values = np.unique(X_mov, return_counts=True)
    print("Unique counts for X_mov:")
    for value, count in zip(unique_counts, unique_values):
        print(f"{value}: {count}")

    # Concatenate movement labels with the extracted features
    X_new_with_movement = np.hstack((X_new, X_mov))

    return X_new_with_movement

# Custom feature extraction functions
def max_func(x):
    return np.max(x, axis=-1)

def min_func(x):
    return np.min(x, axis=-1)

def slope_func(x):
    return np.polyfit(np.arange(x.shape[-1]), x.T, 1)[0]

def median_func(x):
    return np.median(x, axis=-1)

# Save features to a CSV file
def save_features_to_csv(features, output_file):
    feature_df = pd.DataFrame(features)
    feature_df.to_csv(output_file, index=False)
    print(f'Features saved in {output_file}')

def calculate_features(subject_no):
    RAW_CSV_FILE_NAME = f'Filtering\data\\raw_data_sorted\ABCII000{subject_no}_Session_1_rawData_filtered.csv'
    EXTRACTED_FEATURES_FILE = f'Filtering\data\extracted_features\extracted_features_{subject_no}.csv'
    CORRELATION_MATRIX_FILE = f'Filtering\data\correlation_matrices\correlation_matrix_{subject_no}.csv'

    filtered_eeg_raw = preprocess_data(RAW_CSV_FILE_NAME, EEG_COLS, MOVEMENT_COL, SEPARATOR, FS, LOW_CUTOFF, HIGH_CUTOFF)

    events = np.array([[i, 0, 1] for i in range(0, filtered_eeg_raw.n_times - int(WINDOW_SIZE * FS), int(WINDOW_SIZE * (1 - OVERLAP) * FS))]) 

    event_id = {'segment': 1}

    segmented_data = segment_data(filtered_eeg_raw, WINDOW_SIZE, events, event_id)

    features = mne_extract_features(segmented_data)
    #print(features)

    save_features_to_csv(features, EXTRACTED_FEATURES_FILE)

    # Calls the function in the file "eeg_feature_selection.py"
    calculate_corr_coeff(EXTRACTED_FEATURES_FILE, CORRELATION_MATRIX_FILE)

