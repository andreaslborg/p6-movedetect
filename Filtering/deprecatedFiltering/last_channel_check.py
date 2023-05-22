import pandas as pd

def count_ones_in_columns(file_path, col_range):
    df = pd.read_csv(file_path, header=0)  # Set header to 0 to use first row as column labels
    columns_to_check = [str(x) for x in range(col_range[0], col_range[1] + 1)]

    ones_count = 0
    for column in columns_to_check:
        if column in df.columns:
            ones_count += df[column].eq(1).sum()
        else:
            print(f"Column '{column}' not found in the DataFrame")
    
    return ones_count

file_path = "RELEARNBackEnd\Processing\Filtering\data\extracted_features\last_channel.csv"
col_range = (975, 1000)
ones_count = count_ones_in_columns(file_path, col_range)

print(f"Number of 1's in columns 975-1000: {ones_count}")
