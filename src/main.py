
# Library Imports
import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

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
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss=CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.01))
model.fit(x, y, batch_size=256, epochs=10)
model.save("../Models/RedHotLyrics_v1.keras")

# Load trained model
trained_model = load_model("../Models/RedHotLyrics_v1.keras")

# Predict
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = np.random.randint(0, len(input_seqs) - SEQUENCE_LENGTH - 1)
    generated = ['']
    sentence = input_seqs[start_index: start_index + SEQUENCE_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQUENCE_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            for c in character:
                x[0, t, char_to_index[c]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + list(next_character)
    return generated

print(generate_text(100, 0.35))
