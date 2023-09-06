


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflowjs as tfjs

df = pd.read_csv('data.csv')

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']
labels_onehot = pd.get_dummies(labels)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=2)

# Build a neural network model using Keras
model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(7,), activation='relu'),  # Input shape corrected to (7,)
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(labels_onehot.shape[1], activation='softmax')  # Output units match the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))

# Save the model in TensorFlow.js format
tfjs.converters.save_keras_model(model, 'tfjs_model')
