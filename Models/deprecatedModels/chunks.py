import csv
import pandas as pd
import numpy as np

# Define input and output file paths
input_file = 'Data/bisgaard_filtered_eeg_signal.csv'
output_file = 'Data/Bisgaard_filtered_eeg_signal_chunks.csv'

# Define chunk size and initialize row counter
chunk_size = 20
columns_chosen = 1
row_counter = 0



# Open input and output files
with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out, open('Data/chunkTemp.csv', 'w', newline='') as f_temp, open('Data/dataDropped.csv', 'w', newline='') as f_dropped:
    # Create csv reader and writer objects
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)

    # Loop through input rows
    current_chunk = []
    for row in reader:
        # Increment row counter
        row_counter += 1
        
        # Append row to current chunk
        current_chunk.append(row)

        # If chunk is complete, write to output file and reset chunk and row counters
        if row_counter == chunk_size:
            # Check last row for Movement value
            last_movement = int(current_chunk[-1]["Movement"]) == 1

            # Create output row with Movement value
            output_row = [last_movement] + [cell for row in current_chunk for cell in row.values() if cell != "1" and cell != "0"]

            # Write output row to file
            writer.writerow(output_row)

            # Reset chunk and row counters
            current_chunk.remove(current_chunk[0])
            row_counter = chunk_size - 1
    df = pd.read_csv(output_file)
    new_columns = ['Movement'] + list(range(1, (chunk_size*columns_chosen+1)))
    df.columns = new_columns

    df.to_csv(output_file, index=False)