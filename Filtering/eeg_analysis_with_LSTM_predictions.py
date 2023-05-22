import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

df = pd.read_csv('Filtering\\filtered\ABCII0006_filteredData.csv', sep=',',usecols=['Raw Channel Z','Movement'])
pr = pd.read_csv('Models\predictions_un_tr_1235_t_6.csv', sep=',', usecols=['True_label','Predicted_label'])

print(f'df.size = {df.size}')
print(f'pr.size = {pr.size}')
window_size = df.size / pr.size

predicted_movement_df = pr[pr['Predicted_label'] == True]
for index, row in predicted_movement_df.iterrows():
    plt.axvline(x=index*window_size, color='b', linestyle='-', linewidth=1, alpha=0.2)

predicted_movement_df = pr[pr['True_label'] == 1.0]
for index, row in predicted_movement_df.iterrows():
    plt.axvline(x=index*window_size, color='b', linestyle='-', linewidth=1, alpha=1)


# Add a vertical line for rows with 'Movement' value of 1
movement_df = df[df['Movement'] == 1]
for index, row in movement_df.iterrows():
    plt.axvline(x=index, color='r', linestyle='-', linewidth=2)

movement_added = df[df['Movement'] == 2]
for index, row in movement_added.iterrows():
    plt.axvline(x=index, color='g', linestyle='-', linewidth=2)

wrong_movement = df[df['Movement'] == 3]
for index, row in wrong_movement.iterrows():
    plt.axvline(x=index, color='b', linestyle='-', linewidth=2)

movement_moved = df[df['Movement'] == 4]
for index, row in movement_moved.iterrows():
    plt.axvline(x=index, color='y', linestyle='-', linewidth=2)

eeg_signal = df.values.T

# Plot the scattered graph
raw_channel_plots = []
for i in range(eeg_signal.shape[0]-1):  # assuming eeg_signal is a 2D array
    raw_channel_plots.append(plt.plot(range(eeg_signal.shape[1]), eeg_signal[i, :], color='black', label=f'Channel {i+1}'))

# Create legend entries
legend_line_1 = mlines.Line2D([], [], color='r', linestyle='-', linewidth=2, label='1')
legend_line_2 = mlines.Line2D([], [], color='g', linestyle='-', linewidth=2, label='2')
legend_line_3 = mlines.Line2D([], [], color='b', linestyle='-', linewidth=2, label='3')
legend_line_4 = mlines.Line2D([], [], color='y', linestyle='-', linewidth=2, label='4')

plt.ticklabel_format(axis='x', style='plain')
plt.Axes.format_coord = lambda self, x, y: f'x={int(x)}, y={int(y)}'
# Add axis labels and the legend for Raw Channels and vertical lines
plt.xlabel('Time')
plt.ylabel('Filtered EEG')
#plt.legend(handles=[*raw_channel_plots, legend_line_1, legend_line_2, legend_line_3, legend_line_4], loc='upper right')

# Show the plot
plt.show()
