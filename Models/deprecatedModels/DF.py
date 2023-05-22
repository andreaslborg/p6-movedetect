# Import TensorFlow and TensorFlow Decision Forests
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Import pandas for data manipulation
import pandas as pd

# Import NumPy for array manipulation
import numpy as np


# Load the data set into a pandas DataFrame with specified data types and columns
data = pd.read_csv('Data/Bisgaard_chunks.csv')

# Fill missing values with 0
data.fillna(0, inplace=True)

# Convert the "Movement" column to boolean type
data["Movement"] = data["Movement"].astype(bool)

# Split the data into training and testing sets
train_data = data.iloc[:110000]
test_data = data.iloc[110000:]

# Convert the training DataFrame into a TensorFlow dataset
train_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(data, label="Movement")
test_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(data, label="Movement")

# Create a Random Search tuner with 50 trials and automatic hp configuration.
tuner = tfdf.tuner.RandomSearch(num_trials=2, use_predefined_hps=True)

# Define and train the model.
model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
model.fit(train_tf_dataset, verbose=2, class_weight={0: 1, 1: 200})

# Convert the test DataFrame into a NumPy array without labels
test_numpy_array = test_data.drop("Movement", axis=1)

#Compiles model and tests it
model.compile(metrics=[tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.Precision(), tf.keras.metrics.TruePositives()])
evaluation = model.evaluate(test_tf_dataset, return_dict=True)
print()

#Summary of test
for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

#Saves a plot of the decision forest
with open("plot.html", "w") as f: f.write(tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=20))

#Gives an extensive summary of the model
#model.summary()

#Saves the model
model.save("model/superModel")