import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv('RELEARNBackEnd/Processing/Filtering/data/ABCII0001_Session_1_rawData.csv', sep=';')

# Add a vertical line for rows with 'Movement' value of 1
movement_df = df[df['Movement'] == 1]
for index, row in movement_df.iterrows():
    plt.axvline(x=row['Timestamp'], color='r', linestyle='-')

movement_added = df[df['Movement'] == 2]
for index, row in movement_added.iterrows():
    plt.axvline(x=row['Timestamp'], color='g', linestyle='-')

movement_removed = df[df['Movement'] == 3]
for index, row in movement_removed.iterrows():
    plt.axvline(x=row['Timestamp'], color='b', linestyle='-')

movement_removed = df[df['Movement'] == 4]
for index, row in movement_removed.iterrows():
    plt.axvline(x=row['Timestamp'], color='b', linestyle='-')

# Plot the scattered graph
raw_channel_plots = []
raw_channel_plots.append(plt.scatter(df['Timestamp'], df['Raw EMG'], label='Raw EMG'))

plt.ticklabel_format(axis='x', style='plain')
# Add axis labels and the legend for Raw Channels only
plt.xlabel('Timestamp')
plt.ylabel('Raw EMG')
plt.legend(handles=raw_channel_plots, loc='upper right')

# Show the plot
plt.show()

