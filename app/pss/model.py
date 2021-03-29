import re
import csv
import math
import numpy as np
import sklearn.metrics as sklm

from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import *
from keras.utils import *


NB_EMBEDDING_DIMS = 300
NB_SEQUENCE_LENGTH = 150
LABEL2IDX = {'FirstPage' : 1, 'NextPage' : 0}


def simple_tokenizer(textline):
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return words


def read_csv_data(csvfile):
    data_instances = []
    prev_page_text = ""

    # CSV Columns: "counter";"currentText";"label";"documentName"
    with open(csvfile, 'r', encoding='UTF-8') as f:
        datareader = csv.reader(f, delimiter=';', quotechar='"')
        next(datareader)
        for counter, csv_row in enumerate(datareader):
            data_instances.append([csv_row[0], csv_row[1], prev_page_text, csv_row[2]])
            prev_page_text = csv_row[1]
        return data_instances


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


class TextFeatureGenerator(Sequence):
    def __init__(self, text_data, prevpage=False, train=False, batch_size=32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.prevpage = prevpage
        self.train = train

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.prevpage:
            batch_x = self.process_text_data_prevpage(inds)
        else:
            batch_x = self.process_text_data(inds)

        if self.train:
            batch_y = self.generate_output_labels(inds)
        else:
            batch_y = np.zeros(int(len(inds)))

        return batch_x, batch_y

    def process_text_data(self, inds):
        word_embeddings = []
        for index in inds:
            word_embeddings.append(text_to_embedding(self.text_data[index][1]))
        return [np.array(word_embeddings)]

    def process_text_data_prevpage(self, inds):
        word_embeddings = []
        prev_embeddings = []
        for index in inds:
            word_embeddings.append(text_to_embedding(self.text_data[index][1]))
            prev_embeddings.append(text_to_embedding(self.text_data[index][2]))
        return [np.array(word_embeddings), np.array(prev_embeddings)]

    def generate_output_labels(self, inds):
        output_labels = []
        for index in inds:
            temp_output = LABEL2IDX[self.text_data[index][3]]
            output_labels.append(temp_output)
        return np.array(output_labels)


def predict(model, data, prev_page_generator=False, batch_size=256):
    y_predict = np.round(model.predict(TextFeatureGenerator(data, prevpage=prev_page_generator, batch_size=batch_size)))
    return y_predict


def predict_without_rounding(model, data, prev_page_generator=False, batch_size=256):
    y = model.predict(TextFeatureGenerator(data, prevpage=prev_page_generator, batch_size=batch_size))
    return y


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


class ValidationCheckpoint(Callback):
    def __init__(self, filepath, test_data, prev_page_generator=False, metric='kappa'):
        self.test_data = test_data
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        self.prev_page_generator = prev_page_generator

    def on_epoch_end(self, epoch, logs={}):
        
        predicted_labels = predict(self.model, self.test_data, self.prev_page_generator)
        true_labels = [LABEL2IDX[x[3]] for x in self.test_data]

        eval_metrics = {
            'accuracy' : sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro' : sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro' : sklm.f1_score(true_labels, predicted_labels, average='macro'),
            'f1_binary' : sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label=1),
            'kappa' : sklm.cohen_kappa_score(true_labels, predicted_labels)
        }
        eval_metric = eval_metrics[self.metric]
        self.history.append(eval_metric)
        
        if epoch > -1 and eval_metric > self.max_metric:
            print("\n" + self.metric + " improvement: " + str(eval_metric) + " (before: " + str(self.max_metric) + "), saving to " + self.filepath)
            self.max_metric = eval_metric     # optimization target
            self.max_metrics = eval_metrics   # all metrics
            self.model.save(self.filepath)
