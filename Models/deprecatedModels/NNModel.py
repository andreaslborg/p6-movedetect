import tensorflow as tf
from tensorflow.python.keras import layers
import pandas as pd
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# Importing the dataset
csvdata = pd.read_csv('Data/Bisgaard_chunks.csv')
df = pd.DataFrame(csvdata, columns=['Movement', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50'])

# convert boolean column to integer
df['Movement'].fillna(0, inplace=True)
df['Movement'] = df['Movement'].astype(int)

# Splitting the data into training and testing sets
train_df = df.iloc[:150000]
test_df = df.iloc[150000:]

# Separating the features and labels for the training set
train_features = train_df.drop('Movement', axis=1)
train_labels = train_df['Movement']

# Converting the training features and labels to numpy arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)
#train_labels = tf.keras.utils.to_categorical(train_labels)

# Separating the features and labels for the testing set
test_features = test_df.drop('Movement', axis=1)
test_labels = test_df['Movement']

# Converting the testing features and labels to numpy arrays
test_features = np.array(test_features)
test_labels = np.array(test_labels)

#test_labels = tf.keras.utils.to_categorical(test_labels)
neg, pos = np.bincount(df['Movement'])

start = time.perf_counter()
# Setting up the neural network and its different layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu', input_shape=(train_features.shape[1],)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


# Specifying how the model should be compiled
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.Precision(), tf.keras.metrics.Accuracy(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()])

class_weight = {0: 1, 1: 10000}

# Training the model
batch_size = 256
model.fit(train_features, train_labels, epochs=5, class_weight=class_weight, 
          use_multiprocessing=True, workers=-4, batch_size=batch_size)

# Evaluating the model on the testing set
test_loss, test_fn, test_fp, test_precision, test_accuracy, test_tp, test_tn = model.evaluate(test_features, test_labels)
print(f'Test loss: {test_loss}, Test FN: {test_fn}, Test FP: {test_fp},  Test precision: {test_precision}, Test accuracy: {test_accuracy}, Test TP: {test_tp}, Test TN: {test_tn}')


stop = time.perf_counter()
print(f"Trained the model in {stop - start:0.4f} seconds")