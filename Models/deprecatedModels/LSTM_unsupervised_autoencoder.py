import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from CheckAdjacentRows import compare_labels

# Load and preprocess the data
filepaths = ['RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_1.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_2.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_3.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_4.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_5.csv',]

csvdata = pd.concat([pd.read_csv(fp, sep=',') for fp in filepaths], ignore_index=True)

train_set, test_set = train_test_split(csvdata, test_size=0.3, random_state=42)

no_of_columns = 20

train_df = train_set.iloc[:, :no_of_columns]
test_df = test_set.iloc[:, :no_of_columns]

train_labels = train_set.iloc[:, no_of_columns]
test_labels = test_set.iloc[:, no_of_columns]

train_features = np.array(train_df)
test_features = np.array(test_df)

timesteps = 1

train_features = train_features.reshape(train_features.shape[0], timesteps, train_features.shape[1])
test_features = test_features.reshape(test_features.shape[0], timesteps, test_features.shape[1])

# Build the autoencoder
inputs = Input(shape=(timesteps, no_of_columns))
encoded = LSTM(50, activation='relu', return_sequences=False)(inputs)
encoded = Dropout(0.2)(encoded)
encoded = Dense(25, activation='relu')(encoded)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(50, activation='relu', return_sequences=True)(decoded)
decoded = Dropout(0.2)(decoded)
decoded = TimeDistributed(Dense(no_of_columns))(decoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_features, train_features, epochs=50, batch_size=32, validation_split=0.1)

# Encode the features
encoded_train_features = encoder.predict(train_features)
encoded_test_features = encoder.predict(test_features)

# Cluster the encoded features
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(encoded_train_features)

# Assign cluster labels to the test set
test_cluster_labels = kmeans.predict(encoded_test_features)

# Map cluster labels to movement labels (0 or 1)
cluster_label_mapping = {}
for i in range(n_clusters):
    cluster_labels = train_labels[kmeans.labels_ == i]
    majority_label = cluster_labels.mode()[0]
    cluster_label_mapping[i] = majority_label

mapped_test_labels = np.array([cluster_label_mapping[label] for label in test_cluster_labels])

# Evaluate the predictions
cm = confusion_matrix(test_labels, mapped_test_labels)
print('Confusion matrix:')
print(cm)

tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp)

print(f'Test FN: {fn}, Test FP: {fp}, Test TP: {tp}, Test precision: {precision}')

predictions_df = pd.DataFrame({'True_label': test_labels, 'Predicted_label': mapped_test_labels})
predictions_df.to_csv('RELEARNBackEnd\Processing\Models\predictions_un.csv', index=False)

count = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 1, False)
print(f'Total TP within 50 ms: {count}')
