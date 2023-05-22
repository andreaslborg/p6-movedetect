import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the data
filepaths = ['RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_1.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_2.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_3.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_4.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_5.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_6.csv']

# Load the data
data = pd.concat([pd.read_csv(fp, sep=',') for fp in filepaths], ignore_index=True)

# Preprocess the data
scaler = MinMaxScaler()
data.iloc[:, 0:20] = scaler.fit_transform(data.iloc[:, 0:20])

# Prepare the dataset for LSTM
timesteps = 1 
X = np.array(data.iloc[:, 0:20])
X = X.reshape(X.shape[0], timesteps, X.shape[1])
y = np.array(data['20'])


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate class weights
sample_weights = class_weight.compute_sample_weight('balanced', y_train)


print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y:", np.unique(y))
unique_elements, counts = np.unique(y_train, return_counts=True)
print("Counts of unique values in y:", dict(zip(unique_elements, counts)))

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, 20)))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), sample_weight=sample_weights)

# Validate the model
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5 # Convert probabilities to binary values
acc = np.mean(y_pred == y_test) # Calculate accuracy
print('Accuracy:', acc)

cm = confusion_matrix(y_test, y_pred) # Compute confusion matrix
print('Confusion matrix:')
print(cm)


# Create a DataFrame with the actual and predicted movement values
predictions_df = pd.DataFrame({'Movement_before': y_test, 'Predicted_movement': y_pred.ravel()})
predictions_df.to_csv('RELEARNBackEnd\Processing\Models\predictions_su.csv', index=False)
