import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.lines as mlines

# Read in the CSV file
subject_no = 1
csv_file = f'Filtering/data/raw_data/ABCII000{subject_no}_rawData.csv'
df = pd.read_csv(csv_file, sep=';',usecols=['Raw EMG'])

# Getting the unfiltered movement and timestamp column
unfiltered_movement = pd.read_csv(csv_file, delimiter=';', usecols=['Movement','Timestamp'])

# Add a vertical line for rows with 'Movement' value of 1
movement_df = unfiltered_movement[unfiltered_movement['Movement'] == 1]
for index, row in movement_df.iterrows():
    plt.axvline(x=row['Timestamp'], color='r', linestyle='-', linewidth=2)

movement_added = unfiltered_movement[unfiltered_movement['Movement'] == 2]
for index, row in movement_added.iterrows():
    plt.axvline(x=row['Timestamp'], color='g', linestyle='-', linewidth=2)

wrong_movement = unfiltered_movement[unfiltered_movement['Movement'] == 3]
for index, row in wrong_movement.iterrows():
    plt.axvline(x=row['Timestamp'], color='b', linestyle='-', linewidth=2)

movement_moved = unfiltered_movement[unfiltered_movement['Movement'] == 4]
for index, row in movement_moved.iterrows():
    plt.axvline(x=row['Timestamp'], color='y', linestyle='-', linewidth=2)

emg_signal = df.values.T

# Define bandpass filter parameters
low_cutoff = 100 # Hz
high_cutoff = 200  # Hz
sampling_rate = 500  # Hz
filter_order = 4

# Calculate filter coefficients
nyquist_rate = 0.5 * sampling_rate
low = low_cutoff / nyquist_rate
high = high_cutoff / nyquist_rate

mne_info = mne.create_info(ch_names=['Raw EMG'], sfreq=sampling_rate)
eeg_raw = mne.io.RawArray(emg_signal, info=mne_info)

# IIR butterworth filter
bandpass_filter = eeg_raw.copy().filter(
    low_cutoff, high_cutoff, picks=['Raw EMG'], method='iir')

# MNE object to array
mne_eeg_array = bandpass_filter.get_data()

# Save filtered signal to new CSV file and combine with the unfiltered movement column
mne_eeg_df = pd.DataFrame(mne_eeg_array.T, columns=['Raw EMG'])
combined_data = pd.concat([mne_eeg_df, unfiltered_movement], axis=1)

# Plot the scattered graph
raw_channel_plots = []
raw_channel_plots.append(plt.scatter(unfiltered_movement['Timestamp'], mne_eeg_df, label='Raw EMG'))

# Create legend entries
legend_line_1 = mlines.Line2D([], [], color='r', linestyle='-', linewidth=2, label='1')
legend_line_2 = mlines.Line2D([], [], color='g', linestyle='-', linewidth=2, label='2')
legend_line_3 = mlines.Line2D([], [], color='b', linestyle='-', linewidth=2, label='3')
legend_line_4 = mlines.Line2D([], [], color='y', linestyle='-', linewidth=2, label='4')

plt.ticklabel_format(axis='x', style='plain')
plt.Axes.format_coord = lambda self, x, y: f'x={int(x)}, y={int(y)}'
# Add axis labels and the legend for Raw Channels and vertical lines
plt.xlabel('Timestamp')
plt.ylabel('Raw EMG')
plt.legend(handles=[*raw_channel_plots, legend_line_1, legend_line_2, legend_line_3, legend_line_4], loc='upper right')

# Show the plot
plt.show()
