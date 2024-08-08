
# Library Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Define data file path
this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "../data", "lyrics.txt")

# Read available lyrics
with open(DATA_PATH, 'r') as file_obj:
    data = file_obj.read().replace('\n', '')

print(len(data))