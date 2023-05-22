# Import TensorFlow and TensorFlow Decision Forests
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Import pandas for data manipulation
import pandas as pd

# Import NumPy for array manipulation
import numpy as np

dtypes = {
    '1': 'float64', '2': 'float64', '3': 'float64', '4': 'float64', '5': 'float64', '6': 'float64', '7': 'float64', '8': 'float64', '9': 'float64', '10': 'float64', '11': 'float64', '12': 'float64', '13': 'float64', '14': 'float64', '15': 'float64', '16': 'float64', '17': 'float64', '18': 'float64', '19': 'float64', '20': 'float64', 
}

# Define the columns to read
usecols = list(dtypes.keys())

# Load the data set into a pandas DataFrame with specified data types and columns
test_data = pd.read_csv('Data/Bisgaard_filtered_eeg_signal_chunks.csv', dtype=dtypes, usecols=usecols)

# Load the saved model
loaded_model = tf.keras.models.load_model("model/superModel")

#test_data = test_data.iloc[110000:]

predict_data = tfdf.keras.pd_dataframe_to_tf_dataset(test_data)

# Make predictions on the test dataset
predictions = loaded_model.predict(predict_data)

reeee = 0
meep = 0
# Iterate through the prediction array
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        # Check if the value at position (i, j) is non-zero
        if predictions[i,j] >= 0.5:
            reeee +=1
            # Print the non-zero value and its position
            print(f"Value: {predictions[i,j]} Position: ({i}, {j})")
        else: 
            meep +=1

print(reeee)
print(meep)