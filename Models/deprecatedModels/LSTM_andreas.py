import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the data from the CSV files
def load_data(filepaths):
    csvdata = pd.concat([pd.read_csv(fp, sep=',') for fp in filepaths], ignore_index=True)
    return csvdata

# Callback class to store the loss history for each epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append((logs.get('loss'), logs.get('val_loss')))


# Normalize the data (scales the values to be between 0 and 1)
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

# Train the model
def train_model(model, EPOCHS, X_train, y_train, X_val, y_val):
    loss_history = LossHistory()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=32, callbacks=[loss_history])
    return loss_history



def prepare_data(csvdata):
    no_of_columns = 20

    X = csvdata.iloc[:, :no_of_columns]
    y = csvdata.iloc[:, no_of_columns]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Oversample the minority class
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Normalize the data
    X_train_normalized = normalize_data(X_train_resampled)
    X_val_normalized = normalize_data(X_val)

    # Reshape the input data to have the format (batch_size, steps, input_dim)
    X_train_normalized = X_train_normalized.values.reshape(-1, no_of_columns, 1)
    X_val_normalized = X_val_normalized.values.reshape(-1, no_of_columns, 1)

    # Convert the labels to one-hot encoded vectors
    num_classes = 2
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=num_classes)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    return X_train_normalized, X_val_normalized, y_train_one_hot, y_val_one_hot



# Save the loss history to a file
def save_losses(loss_history):
    with open('epoch_70.txt', 'w') as f:
        f.write('Epoch\tTraining Loss\tValidation Loss\n')
        for idx, (train_loss, val_loss) in enumerate(loss_history.losses):
            f.write(f'{idx + 1}\t{train_loss}\t{val_loss}\n')

# Change the create_model() function to accept input_shape as an argument
def create_model(input_shape):
    model = Sequential([
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.25),
        Conv1D(256, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Evaluate the model's performance on the validation set
def evaluate_model(model, X_val, y_val):
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Val loss: {val_loss}')
    print(f'Val accuracy: {val_accuracy}')

# Save the confusion matrix to a file
def save_confusion_matrix(y_val, y_val_pred):
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    y_val_true_labels = np.argmax(y_val, axis=1)
    conf_matrix = confusion_matrix(y_val_true_labels, y_val_pred_labels)

    with open('confusion_matrix.txt', 'w') as f:
        f.write('Confusion matrix:\n')
        for row in conf_matrix:
            f.write(' '.join(str(x) for x in row))
            f.write('\n')

def main():
    TEST_SIZE = 0.2
    EPOCHS = 2

    filepaths = ['RELEARNBackEnd\\Processing\\Filtering\\data\\extracted_features\\extracted_features_1.csv',
                 'RELEARNBackEnd\\Processing\\Filtering\\data\\extracted_features\\extracted_features_2.csv',
                 'RELEARNBackEnd\\Processing\\Filtering\\data\\extracted_features\\extracted_features_3.csv',
                 'RELEARNBackEnd\\Processing\\Filtering\\data\\extracted_features\\extracted_features_4.csv',
                 'RELEARNBackEnd\\Processing\\Filtering\\data\\extracted_features\\extracted_features_5.csv',]

    csvdata = load_data(filepaths)

    X_train_normalized, X_val_normalized, y_train_one_hot, y_val_one_hot = prepare_data(csvdata)

    # Get the input shape of the data for the CNN model
    input_shape = X_train_normalized.shape[1:]
    
    model = create_model(input_shape)

    loss_history = train_model(model, EPOCHS, X_train_normalized, y_train_one_hot, X_val_normalized, y_val_one_hot)

    #save_losses(loss_history)

    evaluate_model(model, X_val_normalized, y_val_one_hot)

    y_val_pred = model.predict(X_val_normalized)
    save_confusion_matrix(y_val_one_hot, y_val_pred)

if __name__ == '__main__':
    main()
