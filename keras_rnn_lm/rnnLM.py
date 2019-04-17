import warnings
import pandas as pd
import numpy as np
import re
import os
import sys

from keras.utils import get_file
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional, SimpleRNN
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle

RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 2048
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.7
VERBOSE = 0
SAVE_MODEL = True
RNN_CELLS = 128


def make_callbacks(model_name, save=SAVE_MODEL):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks

def make_word_level_model(num_words,
                          embedding_matrix,
                          rnn_cells=128,
                          trainable=False,
                          rnn_layers=1,
                          bi_direc=False):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of RNN cell layers."""

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple RNN layers
    if rnn_layers > 1:
        for i in range(rnn_layers - 1):
            model.add(
                SimpleRNN(
                    rnn_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final RNN cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                SimpleRNN(
                    rnn_cells,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            SimpleRNN(
                rnn_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=TRAIN_FRACTION):
    """Create training and validation features and labels."""

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid

def make_sequences(texts,
                   training_length=50,
                   lower=True,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    """Turn a set of texts into sequences of integers"""

    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit to sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]
    over_idx = [
        i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
    ]

    new_texts = []
    new_sequences = []

    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])

    training_seq = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in new_sequences:

        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length:i + 1]

            # Set the features and label
            training_seq.append(extract[:-1])
            labels.append(extract[-1])

    print(f'There are {len(training_seq)} training sequences.')

    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels

def format_patent(patent):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent

if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    K.tensorflow_backend._get_available_gpus()

    # Read in data
    data = pd.read_csv(
        '../data/neural_network_patent_query.csv', parse_dates=['patent_date'])

    # Extract abstracts
    original_abstracts = list(data['patent_abstract'])
    len(original_abstracts)
    print("data samples: \n", data.head())

    formatted = []
    # Iterate through all the original abstracts
    for a in original_abstracts:
        formatted.append(format_patent(a))

    TRAINING_LENGTH = 50

    filters = '!"%;[\\]^_`{|}~\t\n'
    word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(
        formatted, TRAINING_LENGTH, lower=False, filters=filters)

    print("most frequent words: \n", 
		sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15])

    # Vectors to use
    glove_vectors = '/home/jwq/.keras/datasets/glove.6B.zip'

    # Download word embeddings if they are not present
    if not os.path.exists(glove_vectors):
        glove_vectors = get_file('glove.6B.zip',
                                 'http://nlp.stanford.edu/data/glove.6B.zip')
        os.system(f'unzip {glove_vectors}')

    # Load in unzipped file
    glove_vectors = '/home/jwq/.keras/datasets/glove.6B.100d.txt'
    glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
    print("glove.shape:", glove.shape)
    print("glove sample: \n", glove[0])

    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]
    del glove
    print("vectors: \n", vectors[100])
    print("words: \n",  words[100])


    word_lookup = {word: vector for word, vector in zip(words, vectors)}
    embedding_matrix = np.zeros((num_words, vectors.shape[1]))
    not_found = 0

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    print(f'There were {not_found} words without pre-trained embeddings.')

    embedding_matrix = np.zeros((num_words, len(word_lookup['the'])))

    not_found = 0

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    print(f'There were {not_found} words without pre-trained embeddings.')
    print("embedding matrix shape: ", embedding_matrix.shape)

    # Split into training and validation
    X_train, X_valid, y_train, y_valid = create_train_valid(
        features, labels, num_words)
    X_train.shape, y_train.shape

    sys.getsizeof(y_train) / 1e9

    def check_sizes(gb_min=1):
        for x in globals():
            size = sys.getsizeof(eval(x)) / 1e9
            if size > gb_min:
                print(f'Object: {x:10}\tSize: {size} GB.')

    check_sizes(gb_min=1)

    model = make_word_level_model(
        num_words,
        embedding_matrix=embedding_matrix,
        rnn_cells=RNN_CELLS,
        trainable=True,
        rnn_layers=1)
    print(model.summary())


    model_name = 'train-embeddings-rnn-50'
    model_dir = '../my_models/'

    plot_model(model, to_file=f'{model_dir}{model_name}.png', show_shapes=True)

    callbacks = make_callbacks(model_name)
    model_name = 'train-embeddings-rnn-50'
    callbacks = make_callbacks(model_name)
    model.compile(
        optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n ----------  start training  ---------- \n")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid))
    
    print("\n ----------  training end  ---------- \n")
#    model = load_and_evaluate(model_name, return_model=True)
