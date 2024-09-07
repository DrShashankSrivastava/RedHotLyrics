import os
import numpy as np
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
    input_seqs.append(data[i: i + SEQUENCE_LENGTH])
    next_char.append(data[i + SEQUENCE_LENGTH])

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
