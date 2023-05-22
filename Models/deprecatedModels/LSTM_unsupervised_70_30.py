import pandas as pd
import numpy as np
from keras import Input
from keras.models import Model
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
from CheckAdjacentRows import compare_labels
from confusion_matrix_generator import plot_confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

filepaths = ['RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_1.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_2.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_3.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_5.csv',
             'RELEARNBackEnd\Processing\Filtering\data\extracted_features\extracted_features_6.csv']

data = pd.concat([pd.read_csv(fp, sep=',') for fp in filepaths], ignore_index=True)

no_of_columns = 20
timesteps = 40

df = data.iloc[:, :no_of_columns]
labels = data.iloc[:, no_of_columns].values

features = np.array(df)

overlapping_features = []
for i in range(features.shape[0] - timesteps + 1):
    overlapping_features.append(features[i:i+timesteps])
features = np.array(overlapping_features)

labels = labels[timesteps-1:]

# Split the data into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.15, shuffle=True)

# The rest of your code starts here
# Define the LSTM network
inputs = Input(shape=(timesteps, no_of_columns))
x = LSTM(64, activation='elu', dropout=0.2, return_sequences=True)(inputs)
x = LSTM(128, activation='relu', dropout=0.2, return_sequences=True)(x)
x = LSTM(128, activation='tanh', dropout=0.1, return_sequences=True)(x)
x = LSTM(128, activation='gelu', return_sequences=True)(x)
x = LSTM(64, activation='tanh', return_sequences=True)(x)
outputs = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_features, train_labels, epochs=5, batch_size=128)

model.save("superModel")

# Extract features from the train and test data
feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
train_features = feature_extractor.predict(train_features)
test_features = feature_extractor.predict(test_features)

# Flatten the features, removes timesteps again to make the data the correct shape for KMeans
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

# Cluster the train features using k-means
kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
kmeans.fit(train_features)

# Get the cluster assignments for the train and test data
train_cluster_assignments = kmeans.predict(train_features)
test_cluster_assignments = kmeans.predict(test_features)

pred = test_cluster_assignments
print('Test unique values:')
print(pd.Series(test_labels).value_counts(dropna=False))
pred_b = pred > 0.5
pred_b = np.squeeze(pred_b)

predictions_df = pd.DataFrame({'True_label': test_labels, 'Predicted_label': pred_b.ravel()})
predictions_df.to_csv('RELEARNBackEnd\Processing\Models\predictions_un.csv', index=False)

cm = confusion_matrix(test_labels, pred_b)
print('Confusion matrix:')
print(cm)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
print(f'Accuracy: {accuracy * 100}')
print(f'Precision: {precision * 100}')


count1 = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 1)
count1extra = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 1, False, True)
print(f'Total TP within 50 ms: {count1}, including duplicates: {count1extra}')
count2 = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 4, True)
count2extra = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 4, True, True)
print(f'Total TP within past 200 ms: {count2}, including duplicates: {count2extra}')
count3 = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 4, False)
count3extra = compare_labels('RELEARNBackEnd\Processing\Models\predictions_un.csv', 4, False, True)
print(f'Total TP within 200 ms: {count3}, including duplicates: {count3extra}')

plot_confusion_matrix(tn, fp, fn, tp)

