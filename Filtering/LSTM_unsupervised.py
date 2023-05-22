import pandas as pd
import numpy as np
from keras import Input
from keras.models import Model
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
from check_adjacent_rows import compare_labels
from sklearn.cluster import KMeans

def train_lstm(train1, train2, train3, train4, test1, epochs, no_of_features, no_of_channels):
    print('--------------------')
    print(f'Training on participants {train1}, {train2}, {train3}, {train4} and testing on {test1} for {epochs} epochs.')
    print('--------------------')
    filepaths = [f'Filtering\data\extracted_features\extracted_features_{train1}.csv',
                 f'Filtering\data\extracted_features\extracted_features_{train2}.csv',
                 f'Filtering\data\extracted_features\extracted_features_{train3}.csv',
                 f'Filtering\data\extracted_features\extracted_features_{train4}.csv']

    train_data = pd.concat([pd.read_csv(fp, sep=',') for fp in filepaths], ignore_index=True)
    test_data = pd.read_csv(f'Filtering\data\extracted_features\extracted_features_{test1}.csv')

    no_of_columns = no_of_features * no_of_channels
    timesteps = 40

    #Setup the data
    train_df = train_data.iloc[:, :no_of_columns]
    test_df = test_data.iloc[:, :no_of_columns]

    train_labels = train_data.iloc[:, no_of_columns].values
    test_labels = test_data.iloc[:, no_of_columns].values

    train_features = np.array(train_df)
    test_features = np.array(test_df)

    #Sliding windows
    overlapping_train_features = []
    for i in range(train_features.shape[0] - timesteps + 1):
        overlapping_train_features.append(train_features[i:i+timesteps])
    train_features = np.array(overlapping_train_features)

    overlapping_test_features = []
    for i in range(test_features.shape[0] - timesteps + 1):
        overlapping_test_features.append(test_features[i:i+timesteps])
    test_features = np.array(overlapping_test_features)

    train_labels = train_labels[timesteps-1:]
    test_labels = test_labels[timesteps-1:]

    #Define the LSTM
    inputs = Input(shape=(timesteps, no_of_columns))
    x = LSTM(64, activation='elu', dropout=0.2, return_sequences=True)(inputs)
    x = LSTM(128, activation='relu', dropout=0.2, return_sequences=True)(x)
    x = LSTM(128, activation='tanh', dropout=0.1, return_sequences=True)(x)
    x = LSTM(128, activation='gelu', return_sequences=True)(x)
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    #Define the model
    model = Model(inputs=inputs, outputs=outputs)

    #Compile and train the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_features, train_features, epochs=epochs, batch_size=128)

    #Save the model
    model.save("Models\superModel")

    #Extract features from the train and test data
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    train_features = feature_extractor.predict(train_features)
    test_features = feature_extractor.predict(test_features)

    #Flatten the features
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)

    #Train the kmeans using the train features
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
    kmeans.fit(train_features)

    #Get the cluster predictions using k-means
    pred = kmeans.predict(test_features)

    #Get unique values and making prediction into bool
    print('Test unique values:')
    print(pd.Series(test_labels).value_counts(dropna=False))
    pred_b = pred > 0.5
    pred_b = np.squeeze(pred_b)

    #Make csv file with predictions and the true labels
    compare_csv = 'Models\predictions\predictions_un.csv'
    predictions_df = pd.DataFrame({'True_label': test_labels, 'Predicted_label': pred_b.ravel()})
    predictions_df.to_csv(compare_csv, index=False)

    #Generate confusion matrix
    cm = confusion_matrix(test_labels, pred_b)
    print('Confusion matrix:')
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    #Calculate precision and accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    print(f'Accuracy: {accuracy * 100}')
    print(f'Precision: {precision * 100}')

    #Adjacent row checker
    count1 = compare_labels(compare_csv, 1)
    count1extra = compare_labels(compare_csv, 1, False, True)
    print(f'Total TP within 50 ms: {count1}, including duplicates: {count1extra}')
    count2 = compare_labels(compare_csv, 4, True)
    count2extra = compare_labels(compare_csv, 4, True, True)
    print(f'Total TP within past 200 ms: {count2}, including duplicates: {count2extra}')
    count3 = compare_labels(compare_csv, 4, False)
    count3extra = compare_labels(compare_csv, 4, False, True)
    print(f'Total TP within 200 ms: {count3}, including duplicates: {count3extra}')

    return tn, fp, fn, tp
