'''Initialization, import, file name and reading CSV'''
from scipy import signal
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

# Load EEG signal from CSV file
CSV_FILE = 'data/BisgaardData_FixedMovements.csv'
SEPARATOR = ';'
ch_names = ['Raw Channel Z']
#'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four'
eeg_data = pd.read_csv(CSV_FILE, delimiter=SEPARATOR, usecols=ch_names)
eeg_signal = eeg_data.values.T # Transpose and get a numpy array

emg_data = pd.read_csv(CSV_FILE, delimiter=SEPARATOR, usecols=['Raw EMG', 'Raw EMG Reference'])
emg_signal = emg_data.values.T # Transpose and get a numpy array

# Getting the unfiltered movement and timestamp column
unfiltered_movement = pd.read_csv(CSV_FILE, delimiter=SEPARATOR, usecols=['Movement','Timestamp'])

# Filter the rows where Selected is 1 and get the Timestamp column as a numpy array
timestamp_array = unfiltered_movement[unfiltered_movement['Movement']==1]['Timestamp'].to_numpy()
print(f'Timestamp array: {timestamp_array}')

# EEG filtering with MNE

LOW_CUTOFF = 0.1 # Hz
HIGH_CUTOFF = 4 # Hz
FS = 500  # Sampling rate in Hz
# Create MNE info
mne_info = mne.create_info(ch_names=ch_names, sfreq=FS)

# Create MNE object
eeg_raw = mne.io.RawArray(eeg_data.T, info=mne_info)

# Apply bandpass filter using iir method (by default 4th order butterworth filter)
bandpass_filter = eeg_raw.copy().filter(LOW_CUTOFF, HIGH_CUTOFF, picks=ch_names, method='iir')

# Apply notch filter using fir method (by default hamming window)
notch_filter = bandpass_filter.copy().notch_filter(freqs=[50,100,150,200], picks=ch_names)

# Convert the MNE object into an array
mne_eeg_array = notch_filter.get_data()

# Compute PSD using Welch's method for the raw EEG signal
f1, Pxx1 = signal.welch(eeg_signal, FS)
Pxx1 = np.squeeze(Pxx1)

# Compute PSD using Welch's method for the filtered EEG signal
f2, Pxx2 = signal.welch(mne_eeg_array, FS)
Pxx2 = np.squeeze(Pxx2)

# Add the flitered data
filtered_data = pd.DataFrame(mne_eeg_array.T, columns=ch_names)
#'Raw Channel One', 'Raw Channel Two', 'Raw Channel Three', 'Raw Channel Four',
# Save filtered signal to new CSV file and combine with the unfiltered movement column
FILE_NAME = 'filtered/filtered_eeg_signal.csv'
combined_data = pd.concat([filtered_data, unfiltered_movement], axis=1)
combined_data.to_csv(FILE_NAME, index=False)
print(f'CSV saved as {FILE_NAME}')

# MRCP - Find movement indices in the combined data file and add the index to an array
movement_index_array = np.where(combined_data['Movement']==1)[0]
print(f'Movement indices: {movement_index_array}')

# Initialize an empty list to store the extracted values
extracted_values = []
# 1 sec before movement onset and 1 sec after movement onset
EPOCHS_BEFORE = 1000
EPOCHS_AFTER = 1001

# Scaling based on the hardware used so we find the correct microvolt
ADC_BIT = 65536 # 2^16 (16 bit ADC)
ADC_VOLT = 3.3
EEG_GAIN = 50000
VOLT_TO_MU = 1000000 # volt to microvolt

# Loop through movement_index_array
# 16-bit Analog-Digital Converter with 0-3.3V (2^16 = 65536)
for i in movement_index_array:
    # Define index range
    index_range = list(range(i-EPOCHS_BEFORE,i+EPOCHS_AFTER))
    # Extract values from Raw Channel Z using numpy.take
    values = np.take((((combined_data[ch_names[0]]/ADC_BIT)*ADC_VOLT)
                      /EEG_GAIN)*VOLT_TO_MU,index_range)
    # Append values to extracted_values list
    extracted_values.append(values)

# Convert extracted_values list into a numpy array
extracted_values_array = np.array(extracted_values)
print(f'Extracted values array: {extracted_values_array}')

# Calculate the mean value for each column in extracted_values_array
mean_values = np.mean(extracted_values_array, dtype=np.float64, axis=0)
print(f'mean_values array size: {mean_values.size}')

# Plotting graphs

fix, axs = plt.subplots(2,3)

# Adding movement line to the graphs
movement_df = combined_data[combined_data['Movement'] == 1]

for index, row in movement_df.iterrows():
    # Add a vertical line
    axs[0,0].axvline(x=row['Timestamp'], color='r', linestyle='-')
    axs[1,0].axvline(x=row['Timestamp'], color='r', linestyle='-')
    axs[0,2].axvline(x=row['Timestamp'], color='r', linestyle='-')

axs[0,0].set_title('RAW EEG')
axs[0,0].plot(unfiltered_movement['Timestamp'], eeg_data) # eeg_data = RAW EEG data

axs[1,0].set_title('Filtered EEG')
axs[1,0].plot(unfiltered_movement['Timestamp'], mne_eeg_array.T) # filtered

axs[0,1].set_title('RAW Power Spectrum Analysis')
axs[0,1].semilogy(f1, Pxx1)

axs[1,1].set_title('Filtered Power Spectrum Analysis')
axs[1,1].semilogy(f2, Pxx2)

axs[1,2].set_title('MRCP')
axs[1,2].plot(mean_values)
axs[1,2].axvline(x=EPOCHS_BEFORE, color='r', linestyle='-') #Movement onset line
axs[1,2].set_xticks([0, 250, 500, 750, 1000, 1251, 1501, 1751, 2001])
axs[1,2].set_xticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '2'])
axs[1,2].set_xlabel('Time (sec)')
axs[1,2].set_ylabel('Amplitude (µV)')

axs[0,2].set_title('RAW EMG')
axs[0,2].plot(unfiltered_movement['Timestamp'], emg_data) # RAW EMG data

# Show the graphs
plt.show()
