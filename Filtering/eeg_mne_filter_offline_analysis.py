"""
EEG processing script for offline analysis.
"""
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal

"""
Constant values and parameters.
"""
SUBJECT_NO = 2      # Change this number to the subject you want to analyse.
FILTERED_CSV_FILE_NAME = f'Filtering/filtered/ABCII000{SUBJECT_NO}_filteredData.csv'
RAW_CSV_FILE_NAME = f'Filtering/data/raw_data_sorted/ABCII000{SUBJECT_NO}_Session_1_rawData_filtered.csv'
EEG_COLS = ['Raw Channel Z', 'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four']
EMG_COLS = ['Raw EMG', 'Raw EMG Reference']
SEPARATOR = ';'
FS = 500
LOW_CUTOFF = 0.01
HIGH_CUTOFF = 5
NOTCH_FREQS = [50, 100, 150, 200]
EPOCHS_BEFORE = 1000
EPOCHS_AFTER = 1001
ADC_BIT = 65536
ADC_VOLT = 3.3
EEG_GAIN = 50000
VOLT_TO_MU = 1000000

def load_data(csv_file, separator, eeg_cols, emg_cols):
    """
    Reads the input CSV file, extracting EEG, EMG, Movement, and Timestamp columns, 
    and returns them as separate DataFrames.
    """
    eeg_data = pd.read_csv(csv_file, delimiter=separator, usecols=eeg_cols)
    eeg_data_df = eeg_data.values.T # dataframe ndarray

    emg_data = pd.read_csv(csv_file, delimiter=separator, usecols=emg_cols)

    unfiltered_movement = pd.read_csv(
        csv_file, delimiter=separator, usecols=['Movement', 'Timestamp'])

    return eeg_data, eeg_data_df, emg_data, unfiltered_movement


def preprocess_data(eeg_data_df, eeg_cols, sampling_rate, low_cutoff, high_cutoff, notch_freqs):
    """
    Filters the input EEG data using a bandpass Butterworth filter. 
    The filtered data is returned as a NumPy array.
    """
    mne_info = mne.create_info(ch_names=eeg_cols, sfreq=sampling_rate, ch_types=['eeg'] * 5)
    eeg_raw = mne.io.RawArray(eeg_data_df, info=mne_info)

    # IIR butterworth filter
    bandpass_filter = eeg_raw.copy().filter(
        low_cutoff, high_cutoff, picks=eeg_cols, method='iir')

    # FIR notch filter (hamming window)
    #notch_filter = bandpass_filter.copy().notch_filter(
    #    freqs=notch_freqs, picks=eeg_cols)

    # MNE object to array
    mne_eeg_array = bandpass_filter.get_data()

    return mne_eeg_array


def compute_psd(eeg_data_df, sampling_rate):
    """
    Computes the power spectral density (PSD) of the input EEG data 
    using the Welch's method. Returns frequency and power values.
    """
    freq, pxx = signal.welch(eeg_data_df, sampling_rate)
    pxx = np.squeeze(pxx)

    return freq, pxx


def extract_mrcp_values(combined_data, eeg_cols, movement_index_array,
                        epochs_before, epochs_after, adc_bit, adc_volt, eeg_gain, volt_to_mu):
    """
    Extracts MRCP values from the input data using movement indices, 
    and returns the mean MRCP values across all epochs.
    """
    extracted_values = []

    # Calculate the mean of all EEG channels
    mean_eeg_data = combined_data[eeg_cols].mean(axis=1)

    for i in movement_index_array:
        if i - epochs_before < 0 or i + epochs_after >= len(mean_eeg_data):
            print(f"Skipping data points from index {i - epochs_before} to {i + epochs_after} as they are out of bounds.")
            continue

        index_range = list(range(i - epochs_before, i + epochs_after))

        values = np.take((((mean_eeg_data / adc_bit) * adc_volt) / eeg_gain) 
                         * volt_to_mu, index_range)

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

    added_movement_df = unfiltered_movement[unfiltered_movement['Movement'] == 2]
    for _, row in added_movement_df.iterrows():
        axs[0, 0].axvline(x=row['Timestamp'], color='b', linestyle='-')
        axs[1, 0].axvline(x=row['Timestamp'], color='b', linestyle='-')
        axs[0, 2].axvline(x=row['Timestamp'], color='b', linestyle='-')

    timestamp = unfiltered_movement['Timestamp']

    axs[0, 0].set_title('RAW EEG')
    axs[0, 0].plot(timestamp, eeg_data, label=EEG_COLS)  
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend(loc='lower right')

    axs[1, 0].set_title('Filtered EEG')
    axs[1, 0].plot(timestamp, mne_eeg_array.T, label=EEG_COLS)  
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend(loc='lower right')
    
    axs[0, 1].set_title('RAW PSD Analysis')
    axs[0, 1].semilogy(freq1, psd1, label=EEG_COLS[0])
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_ylabel('PSD [(ADC Level^2) / Hz]')
    axs[0, 1].legend(loc='upper right')

    axs[1, 1].set_title('Filtered PSD Analysis')
    axs[1, 1].semilogy(freq2, psd2, label=EEG_COLS[0])
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('PSD [(ADC Level^2) / Hz]')
    axs[1, 1].legend(loc='upper right')
    
    axs[1, 2].set_title(f'Subject {SUBJECT_NO} - MRCP')
    axs[1, 2].plot(mean_values)
    axs[1, 2].axvline(x=epochs_before, color='r', linestyle='-')
    axs[1, 2].set_xticks([0, 250, 500, 750, 1000, 1251, 1501, 1751, 2001])
    axs[1, 2].set_xticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2'])
    axs[1, 2].set_xlabel('Time (sec)')
    axs[1, 2].set_ylabel('Amplitude (\u03BCV)')

    axs[0, 2].set_title('RAW EMG')
    axs[0, 2].plot(unfiltered_movement['Timestamp'], emg_data, label=EMG_COLS)
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Amplitude')
    axs[0, 2].legend(loc='lower right')
    
    plt.show()


def save_csv(mne_eeg_array, ch_names, unfiltered_movement):
    # Add the flitered data
    filtered_data = pd.DataFrame(mne_eeg_array.T, columns=ch_names)
    #'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four',
    # Save filtered signal to new CSV file and combine with the unfiltered movement column

    combined_data = pd.concat([filtered_data, unfiltered_movement], axis=1)
    combined_data.to_csv(FILTERED_CSV_FILE_NAME, index=False)
    print(f'CSV saved as {FILTERED_CSV_FILE_NAME}')

def main(eeg_cols, emg_cols, csv_file, separator):
    '''
    The main function that calls the other functions in the script, passing the necessary parameters 
    and arguments for data processing and visualization.
    '''
    # Load and preprocess data
    eeg_data, eeg_data_df, emg_data, unfiltered_movement = load_data(csv_file, separator, eeg_cols, emg_cols)
    mne_eeg_array = preprocess_data(eeg_data_df, eeg_cols, FS, LOW_CUTOFF, HIGH_CUTOFF, NOTCH_FREQS) # NOTCH_FREQS is currently not used.

    #unfiltered_movement['Movement'] = unfiltered_movement['Movement'].replace(3, 0)
    #unfiltered_movement['Movement'] = unfiltered_movement['Movement'].replace(4, 0)

    # Save filtered data to combined_data
    filtered_data = pd.DataFrame(mne_eeg_array.T, columns=eeg_cols)
    combined_data = pd.concat([filtered_data, unfiltered_movement], axis=1)

    save_csv(mne_eeg_array, EEG_COLS, unfiltered_movement)

    # Compute PSD for Raw Channel Z (index 0) raw and filtered
    freq1, psd1 = compute_psd(eeg_data_df[0], FS)
    freq2, psd2 = compute_psd(mne_eeg_array[0], FS)

    # Store indices where Movement = 1 and Movement = 2
    movement_index_array = np.where(
        np.logical_or(
        unfiltered_movement['Movement'] == 1, 
        unfiltered_movement['Movement'] == 2,
        ))[0]
    
    print(f'Total movements: {movement_index_array.size}')

    # Extract MRCP values and calculate the mean
    mean_values = extract_mrcp_values(combined_data, eeg_cols, movement_index_array, 
                                      EPOCHS_BEFORE, EPOCHS_AFTER, ADC_BIT, ADC_VOLT, EEG_GAIN, VOLT_TO_MU)

    # Plot graphs
    plot_graphs(unfiltered_movement, eeg_data, mne_eeg_array, freq1, psd1, freq2, psd2, 
                mean_values, emg_data, EPOCHS_BEFORE)

if __name__ == '__main__':
    main(EEG_COLS, EMG_COLS, RAW_CSV_FILE_NAME, SEPARATOR)
