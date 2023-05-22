"""
EEG processing script
"""

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import signal


def load_data(csv_file, separator, eeg_cols, emg_cols):
    """
    Reads the input CSV file, extracting EEG, EMG, Movement, and Timestamp columns, 
    and returns them as separate DataFrames.
    """
    print('Loading data from csv')
    eeg_data = pd.read_csv(csv_file, delimiter=separator, usecols=eeg_cols)
    eeg_data_df = eeg_data.values.T # dataframe ndarray

    emg_data = pd.read_csv(csv_file, delimiter=separator, usecols=emg_cols)

    unfiltered_movement = pd.read_csv(
        csv_file, delimiter=separator, usecols=['Movement', 'Timestamp'])

    return eeg_data, eeg_data_df, emg_data, unfiltered_movement


def live_data(eeg_data_df):
    """
    Load live EEG data and preprocess every window
    """
    print('Filtering live data')
    windows = []

    for start_index in range(eeg_data_df.shape[1] - WINDOWS_SIZE + 1):
        end_index = start_index + WINDOWS_SIZE
        window = eeg_data_df[:, start_index:end_index]
        window_filtered = preprocess_data(window, EEG_COLS, FS, LOW_CUTOFF, HIGH_CUTOFF, NOTCH_FREQS) 
        window_filtered_flat = window_filtered.flatten() # Flatten the 2D array into a 1D array
        windows.append(window_filtered_flat)

    return windows


def preprocess_data(eeg_data_df, eeg_cols, sampling_rate, low_cutoff, high_cutoff, notch_freqs):
    """
    Filters the input EEG data using a bandpass Butterworth filter and a notch filter. 
    The filtered data is returned as a NumPy array.
    """
    ch_types = ['eeg' for _ in eeg_cols] 
    mne_info = mne.create_info(ch_names=eeg_cols, sfreq=sampling_rate, ch_types=ch_types)
    eeg_raw = mne.io.RawArray(eeg_data_df, info=mne_info)
    # IIR butterworth filter
    bandpass_filter = eeg_raw.copy().filter(
        low_cutoff, high_cutoff, picks=eeg_cols, method='iir')
    # FIR notch filter (hamming window)
    #notch_filter = bandpass_filter.copy().notch_filter(
    #    freqs=notch_freqs, picks=eeg_cols)

    mne_eeg_array = bandpass_filter.get_data()

    return mne_eeg_array

def compute_psd(eeg_data_df, sampling_rate):
    """
    Computes the power spectral density (PSD) of the input EEG data 
    using the Welch's method. Returns frequency and power values.
    """
    freq, psd = signal.welch(eeg_data_df, sampling_rate)
    psd = np.squeeze(psd)

    return freq, psd

def extract_mrcp_values(combined_data, eeg_cols, movement_index_array,
                        epochs_before, epochs_after, adc_bit, adc_volt, eeg_gain, volt_to_mu):
    """
    Extracts MRCP values from the input data using movement indices, 
    and returns the mean MRCP values across all epochs.
    """
    extracted_values = []

    for i in movement_index_array:
        if i - epochs_before >= 0 and i + epochs_after < len(combined_data):
            index_range = list(range(i - epochs_before, i + epochs_after))

            values = np.take((((combined_data[eeg_cols[0]] / adc_bit)
                               * adc_volt) / eeg_gain) * volt_to_mu, index_range)

            extracted_values.append(values)

    extracted_values_array = np.array(extracted_values)

    mean_values = np.mean(extracted_values_array, dtype=np.float64, axis=0)

    return mean_values

def plot_graphs(unfiltered_movement, eeg_data, mne_eeg_array,
                freq1, psd1, freq2, psd2, mean_values, emg_data, epochs_before):
    """
    Plots the raw and filtered EEG, raw and filtered PSD analysis, MRCP, 
    and EMG data in separate subplots for visualization.
    """
    fig, axs = plt.subplots(2, 3)

    # Draw movement lines of these graphs
    movement_df = unfiltered_movement[unfiltered_movement['Movement'] == 1]
    for _, row in movement_df.iterrows():
        axs[0, 0].axvline(x=row['Timestamp'], color='r', linestyle='-')
        axs[1, 0].axvline(x=row['Timestamp'], color='r', linestyle='-')
        axs[0, 2].axvline(x=row['Timestamp'], color='r', linestyle='-')

    axs[0, 0].set_title('RAW EEG')
    axs[0, 0].plot(unfiltered_movement['Timestamp'], eeg_data)

    axs[1, 0].set_title('Filtered EEG')
    axs[1, 0].plot(unfiltered_movement['Timestamp'], mne_eeg_array.T)

    axs[0, 1].set_title('RAW PSD Analysis')
    axs[0, 1].semilogy(freq1, psd1)

    axs[1, 1].set_title('Filtered PSD Analysis')
    axs[1, 1].semilogy(freq2, psd2)

    axs[1, 2].set_title('MRCP')
    axs[1, 2].plot(mean_values)
    axs[1, 2].axvline(x=epochs_before, color='r', linestyle='-')
    axs[1, 2].set_xticks([0, 250, 500, 750, 1000, 1251, 1501, 1751, 2001])
    axs[1, 2].set_xticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2'])
    axs[1, 2].set_xlabel('Time (sec)')
    axs[1, 2].set_ylabel('Amplitude (\u03BCV)')

    axs[0, 2].set_title('RAW EMG')
    axs[0, 2].plot(unfiltered_movement['Timestamp'], emg_data)

    plt.show()

def save_csv(mne_eeg_array, ch_names, unfiltered_movement):
    # Add the flitered data
    filtered_data = pd.DataFrame(mne_eeg_array.T, columns=ch_names)
    #'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four',
    # Save filtered signal to new CSV file and combine with the unfiltered movement column

    combined_data = pd.concat([filtered_data, unfiltered_movement], axis=1)
    combined_data.to_csv(FILTERED_CSV_FILE_NAME, index=False)
    print(f'CSV saved as {FILTERED_CSV_FILE_NAME}')

# Constant values and parameters
FILTERED_CSV_FILE_NAME = 'RELEARNBackEnd/Processing/Filtering/filtered/filtered_bis_eeg.csv'
RAW_CSV_FILE_NAME = 'RELEARNBackEnd\Processing\Filtering\data\BisgaardData_FixedMovements.csv'
EEG_COLS = ['Raw Channel Z']
EMG_COLS = ['Raw EMG', 'Raw EMG Reference']
SEPERATOR = ';'
FS = 500
LOW_CUTOFF = 0.05
HIGH_CUTOFF = 4
NOTCH_FREQS = [50, 100, 150, 200]
EPOCHS_BEFORE = 1000
EPOCHS_AFTER = 1001
ADC_BIT = 65536
ADC_VOLT = 3.3
EEG_GAIN = 50000
VOLT_TO_MU = 1000000
WINDOWS_SIZE = 1000 # Live window filtering size in Hz / samplings

def make_prediction(window, model):
    # Make sure the input window is a NumPy array with the correct shape
    # (assuming your model takes a single window with the shape (1, 100, 1) as input)
    input_data = np.array(window).reshape(1, WINDOWS_SIZE)

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Convert the prediction to True/False
    # Assuming prediction is a single value (probability) and the threshold is 0.5
    return prediction[0][0] > 0.5

def main(eeg_cols, emg_cols, csv_file, separator):
    '''
    The main function that calls the other functions in the script, passing the necessary parameters 
    and arguments for data processing and visualization.
    '''
    # Load and preprocess data
    mne.set_log_level('WARNING')
    eeg_data, eeg_data_df, emg_data, unfiltered_movement = load_data(csv_file, separator, eeg_cols, emg_cols)
    mne_eeg_array = preprocess_data(eeg_data_df, eeg_cols, FS, LOW_CUTOFF, HIGH_CUTOFF, NOTCH_FREQS)

    # Save filtered data to combined_data
    filtered_data = pd.DataFrame(mne_eeg_array.T, columns=eeg_cols)
    combined_data = pd.concat([filtered_data, unfiltered_movement], axis=1)

    save_csv(mne_eeg_array, EEG_COLS, unfiltered_movement)

    # Compute PSD
    freq1, psd1 = compute_psd(eeg_data_df, FS)
    freq2, psd2 = compute_psd(mne_eeg_array, FS)

    # Extract MRCP values
    movement_index_array = np.where(unfiltered_movement['Movement'] == 1)[0]
    mean_values = extract_mrcp_values(combined_data, eeg_cols, movement_index_array, 
                                      EPOCHS_BEFORE, EPOCHS_AFTER, ADC_BIT, ADC_VOLT, EEG_GAIN, VOLT_TO_MU)

    print('Preprocessing live data started')
    windows = live_data(eeg_data_df)
    #print(f'Windows: {windows}')
    print('Preprocessing live data ended')
    model_path = 'RELEARNBackEnd\Processing\Filtering\data\my_model.h5'
    model = load_model(model_path)
    print(f'Predictions saved to {model_path}')

    predictions = []
    
    for window in windows:
        result = make_prediction(window, model)
        predictions.append(result)

    # Create a DataFrame with the predictions
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])

    # Save the DataFrame to a CSV file
    predictions_df.to_csv('RELEARNBackEnd\Processing\Filtering\data\predictions.csv', index=False)


    # Plot graphs
    plot_graphs(unfiltered_movement, eeg_data, mne_eeg_array, freq1, psd1, freq2, psd2, 
                mean_values, emg_data, EPOCHS_BEFORE)


# main(eeg channel, emg channels, location of csv file, separator used)
main(EEG_COLS, EMG_COLS, RAW_CSV_FILE_NAME, SEPERATOR)