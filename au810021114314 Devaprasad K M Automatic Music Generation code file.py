# Automatic Music Generation
# Goal: Develop an end-to-end model for Automatic Music Generation.
# Models: Implement WaveNet and LSTM architectures in Keras, compare their performances, and evaluate their capability to generate music.

# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
from collections import Counter
from music21 import converter, instrument, note, chord, stream
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Conv1D, Dropout, MaxPool1D, GlobalMaxPool1D, Dense, LSTM, Activation
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K

# Step 1: Load and Preprocess MIDI Data
def read_midi(file_path):
    """Reads a MIDI file and extracts notes and chords."""
    print("Loading Music File:", file_path)
    notes = []
    midi = converter.parse(file_path)
    s2 = instrument.partitionByInstrument(midi)
    
    for part in s2.parts:
        if 'Piano' in str(part): 
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return np.array(notes)

# Load dataset of MIDI files
path = '../input/beethoven-midi/'  # Set your dataset path
files = [i for i in os.listdir(path) if i.endswith(".mid")]
notes_array = np.array([read_midi(path + i) for i in files])

# Aggregate notes and filter by frequency
notes_ = [note for note_seq in notes_array for note in note_seq]
freq = dict(Counter(notes_))
frequent_notes = [note_ for note_, count in freq.items() if count >= 50]

# Prepare filtered data for training
new_music = []
for notes in notes_array:
    temp = [note for note in notes if note in frequent_notes]
    new_music.append(temp)
new_music = np.array(new_music)

# Step 2: Prepare Sequential Data for Model Training
no_of_timesteps = 32
x, y = [], []
for note_seq in new_music:
    for i in range(0, len(note_seq) - no_of_timesteps, 1):
        input_seq = note_seq[i:i + no_of_timesteps]
        output_note = note_seq[i + no_of_timesteps]
        x.append(input_seq)
        y.append(output_note)
x, y = np.array(x), np.array(y)

# Encode sequences as integers
unique_x = list(set(x.ravel()))
x_note_to_int = {note: number for number, note in enumerate(unique_x)}
x_seq = np.array([[x_note_to_int[note] for note in seq] for seq in x])

unique_y = list(set(y))
y_note_to_int = {note: number for number, note in enumerate(unique_y)}
y_seq = np.array([y_note_to_int[note] for note in y])

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

# Step 3: Define and Train the WaveNet Model
K.clear_session()
wavenet_model = Sequential([
    Embedding(len(unique_x), 100, input_length=32, trainable=True),
    Conv1D(64, 3, padding='causal', activation='relu'),
    Dropout(0.2),
    MaxPool1D(2),
    Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'),
    Dropout(0.2),
    MaxPool1D(2),
    Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'),
    Dropout(0.2),
    MaxPool1D(2),
    GlobalMaxPool1D(),
    Dense(256, activation='relu'),
    Dense(len(unique_y), activation='softmax')
])

# Compile and train WaveNet model
wavenet_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
mc_wavenet = ModelCheckpoint('wavenet_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history_wavenet = wavenet_model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val), verbose=1, callbacks=[mc_wavenet])
print("WaveNet model training complete.")

# Load best WaveNet model
wavenet_model = load_model('wavenet_model.h5')

# Step 4: Generate Music with WaveNet
ind = np.random.randint(0, len(x_val) - 1)
random_music = x_val[ind]
predictions = []

for i in range(10):
    random_music = random_music.reshape(1, no_of_timesteps)
    prob = wavenet_model.predict(random_music)[0]
    y_pred = np.argmax(prob, axis=0)
    predictions.append(y_pred)
    random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
    random_music = random_music[1:]

# Map predictions to notes
x_int_to_note = {number: note for number, note in enumerate(unique_x)}
predicted_notes = [x_int_to_note[i] for i in predictions]

# Convert predictions to MIDI file
def convert_to_midi(prediction_output):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='wavenet_music.mid')

convert_to_midi(predicted_notes)

# Step 5: Define and Train the LSTM Model
K.clear_session()
lstm_model = Sequential([
    Embedding(len(unique_x), 100, input_length=32, trainable=True),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(256, activation='relu'),
    Dense(len(unique_x), activation='softmax')
])

# Compile and train LSTM model
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
mc_lstm = ModelCheckpoint('lstm_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history_lstm = lstm_model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val), verbose=1, callbacks=[mc_lstm])
print("LSTM model training complete.")

# Load best LSTM model for further evaluation if needed
lstm_model = load_model('lstm_model.h5')

# Summary:
# - We've built and trained two different models (WaveNet and LSTM) for music generation.
# - You can now evaluate each model's output and compare their performance in terms of generated music quality.
# - Use wavenet_music.mid and other generated MIDI files to listen and analyze the results.