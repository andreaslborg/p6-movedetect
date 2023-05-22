import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Data/Bisgaard_filtered_eeg_signal_chunks.csv')

# Fill missing values with 0
data.fillna(0, inplace=True)

# Split the data into features and labels
X = data.drop(columns=['Movement']).values.astype(float)
Y = data['Movement'].values.astype(int)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y = Y*2 -1

# Define the SVM model
model = tf.keras.models.Sequential([
    tf.keras.Input(X_train.shape[1],),
    tf.keras.layers.experimental.RandomFourierFeatures(
    output_dim = 100,
    ),
    tf.keras.layers.Dense(units=1),
    tf.keras.layers.Lambda(lambda x: tf.math.sign(x))
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss=tf.keras.losses.hinge, 
              metrics=[tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.Precision(), tf.keras.metrics.Accuracy(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()])

# Train the model
model.fit(X_train, Y_train, epochs=20, class_weight={0: 1, 1: 100}, verbose = 2)

# Evaluating the model on the testing set
test_loss, test_fn, test_fp, test_precision, test_accuracy, test_tp, test_tn = model.evaluate(X_test, Y_test)
print(f'Test loss: {test_loss}, Test FN: {test_fn}, Test FP: {test_fp},  Test precision: {test_precision}, Test accuracy: {test_accuracy}, Test TP: {test_tp}, Test TN: {test_tn}')