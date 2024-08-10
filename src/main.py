
# Library Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define data file path
this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "../data", "lyrics.txt")

# Read available lyrics
with open(DATA_PATH, 'r') as file_obj:
    data = file_obj.read().replace('\n', '')
data = data.lower()

# Extract all unique characters
characters = sorted(set(data))

# Convert char to int and vice-versa
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Define model input length and step size
SEQUENCE_LENGTH = 40
STEP_SIZE = 5

# Placeholders for inputs and targets
input_seqs = []
next_char = []

# Form inputs and target for model training
for i in range(0, len(data) - SEQUENCE_LENGTH, STEP_SIZE):
    input_seqs.append(data[i:i+SEQUENCE_LENGTH])
    next_char.append(data[i+SEQUENCE_LENGTH])

# Define input and output placeholders
x = np.zeros((len(input_seqs), SEQUENCE_LENGTH, len(characters)), dtype='bool')
y = np.zeros((len(input_seqs), len(characters)), dtype='bool')

# Prepare the input and output for ML
for i, input_seq in enumerate(input_seqs):
    for j, character in enumerate(input_seq):
        x[i, j, char_to_index[character]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Create a Recurring Neural Network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))
model.add(keras.layers.Dense(len(characters)))
model.add(keras.layers.Activation('softmax'))
model.compile(loss=keras.losses.CategoricalCrossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.01))
model.fit(x, y, batch_size=256, epochs=10)
model.save("../Models/RedHotLyrics_v1.keras")
