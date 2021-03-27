import re
import math
import numpy as np

from keras.models import Model
from keras.layers import *
from keras.utils import *


NB_EMBEDDING_DIMS = 300
NB_SEQUENCE_LENGTH = 150


def simple_tokenizer(textline):
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return words


def text_to_embedding(text_sequence):
    temp_word = []

    # tokenize
    sentence = simple_tokenizer(text_sequence)

    # trim to max sequence length
    if len(sentence) > NB_SEQUENCE_LENGTH:
        half_idx = int(NB_SEQUENCE_LENGTH / 2)
        tmp_sentence = sentence[:half_idx]
        tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
        sentence = tmp_sentence

    # padding
    words_to_pad = NB_SEQUENCE_LENGTH - len(sentence)
    for i in range(words_to_pad):
        sentence.append('PADDING_TOKEN')

    # create data input for words
    for w_i, word in enumerate(sentence):
        if word == 'PADDING_TOKEN':
            word_vector = [0] * NB_EMBEDDING_DIMS
        else:
            word_vector = ft.get_word_vector(word.lower())
        temp_word.append(word_vector)

    return temp_word


class TextFeatureSinglePageGenerator(Sequence):
    def __init__(self, text_data, batch_size=32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):
        word_embeddings = []
        for index in inds:
            word_embeddings.append(text_to_embedding(self.text_data[index][1]))
        return [np.array(word_embeddings)], np.zeros(int(len(inds)))


class TextFeaturePrevPageGenerator(Sequence):
    def __init__(self, text_data, batch_size=32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):
        word_embeddings = []
        prev_embeddings = []
        for index in inds:
            word_embeddings.append(text_to_embedding(self.text_data[index][1]))
            prev_embeddings.append(text_to_embedding(self.text_data[index][2]))
        return [np.array(word_embeddings), np.array(prev_embeddings)], np.zeros(int(len(inds)))


def predict(model, data, prev_page_generator=False, batch_size=256):
    if prev_page_generator:
        y_predict = np.round(model.predict(TextFeaturePrevPageGenerator(data, batch_size=batch_size)))
    else:
        y_predict = np.round(model.predict(TextFeatureSinglePageGenerator(data, batch_size=batch_size)))
    return y_predict


def compile_model_singlepage(print_summary=False):
    filter_sizes = (3, 4, 5)
    model_input_ft = Input(shape=(NB_SEQUENCE_LENGTH, NB_EMBEDDING_DIMS))
    gru_block = Bidirectional(GRU(128, dropout=0.5, return_sequences=True))(model_input_ft)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters=200,
            kernel_size=sz,
            padding="same",
            strides=1
        )(gru_block)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks.append(conv)
    model_concatenated = concatenate(conv_blocks)
    model_concatenated = Dense(128)(model_concatenated)
    model_concatenated = LeakyReLU()(model_concatenated)
    model_output = Dense(1, activation="sigmoid")(model_concatenated)
    model = Model([model_input_ft], model_output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    if print_summary:
        model.summary()
    return model


def compile_model_prevpage(print_summary=False):
    filter_sizes = (3, 4, 5)
    model_input_tp = Input(shape=(NB_SEQUENCE_LENGTH, NB_EMBEDDING_DIMS))
    gru_block_tp = Bidirectional(GRU(128, dropout=0.5, return_sequences=True))(model_input_tp)
    conv_blocks_tp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters=200,
            kernel_size=sz,
            padding="same",
            strides=1
        )(gru_block_tp)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks_tp.append(conv)
    model_concatenated_tp = concatenate(conv_blocks_tp)
    model_concatenated_tp = Dense(128)(model_concatenated_tp)
    model_concatenated_tp = LeakyReLU()(model_concatenated_tp)

    model_input_pp = Input(shape=(NB_SEQUENCE_LENGTH, NB_EMBEDDING_DIMS))
    gru_block_pp = Bidirectional(GRU(128, dropout=0.5, return_sequences=True))(model_input_pp)
    conv_blocks_pp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters=200,
            kernel_size=sz,
            padding="same",
            strides=1
        )(gru_block_pp)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks_pp.append(conv)
    model_concatenated_pp = concatenate(conv_blocks_pp)
    model_concatenated_pp = Dense(128)(model_concatenated_pp)
    model_concatenated_pp = LeakyReLU()(model_concatenated_pp)

    # concat both + another dense
    page_sequence = concatenate([model_concatenated_tp, model_concatenated_pp])
    page_sequence = Dense(256)(page_sequence)
    page_sequence = LeakyReLU()(page_sequence)

    # prediction layer
    model_output = Dense(1, activation="sigmoid")(page_sequence)

    # combine final model
    model = Model([model_input_tp, model_input_pp], model_output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    if print_summary:
        model.summary()
    return model
