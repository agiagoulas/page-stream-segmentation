import csv, re, math
import sklearn.metrics as sklm
import fasttext
import numpy as np

from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import *
from keras.callbacks import ModelCheckpoint, Callback


nb_embedding_dims = 300 # ft.get_dimension()
nb_sequence_length = 150
word_vectors_ft = {}
label2Idx = {'FirstPage' : 1, 'NextPage' : 0}

def simple_tokenizer(textline):
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    # print(words)
    return(words)


def read_csv_data(csvfile, csvformat = "archive20k", return_DC = False):
    
    data_instances = []
    instance_ids = []
    current_id = 0
    
    prevBinder = ""
    prevPageText = ""
    prevPageClass = ""
    
    with open(csvfile, 'r', encoding='UTF-8') as f:
        datareader = csv.reader(f, delimiter=';', quotechar='"')
        next(datareader)
        for instance in datareader:
            # "binder";"docid";"class";"type";"text"
            if prevBinder == instance[0]:
                prevPage = prevPageText
            else:
                prevPage = ""

        
            # 0 - prevbinder - binder - filename ?
            # 1 - docid - some id ? 
            # 2 - prevpageclass - class (FirstPage or NextPage)
            # 3 - prevpagetext - Format ?
            # 4 - ? 

            # data_instances (docid, class, text, prevpagetext (if prevpage), binder)

                
            if csvformat == "Tobacco800":
                data_instances.append([instance[1], instance[2], instance[3], prevPage, instance[0]])
                prevBinder = instance[0]
                prevPageText = instance[3]
                prevPageClass = instance[2]
            else:
                if return_DC:
                    if instance[2] == "FirstPage" or prevPageClass == "FirstPage":
                        data_instances.append([instance[1], instance[2], instance[4], prevPage, instance[0], instance[3]])
                        instance_ids.append(current_id)
                else:
                    data_instances.append([instance[1], instance[2], instance[4], prevPage, instance[0]])
                prevBinder = instance[0]
                prevPageText = instance[4]
                prevPageClass = instance[2]
                
            current_id += 1
    if len(instance_ids) > 0:
        return data_instances, instance_ids
    else:
        return data_instances



class TextFeatureGenerator(Sequence):
    def __init__(self, text_data, batch_size = 32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):

        word_embeddings = []
        output_labels = []

        for index in inds:
            
            temp_word = []
            
            # tokenize
            sentence = simple_tokenizer(self.text_data[index][2])
            temp_output = label2Idx[self.text_data[index][1]]
            
            # trim to max sequence length
            if (len(sentence) > nb_sequence_length):
                half_idx = int(nb_sequence_length / 2)
                tmp_sentence = sentence[:half_idx]
                tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
                sentence = tmp_sentence

            # padding
            words_to_pad = nb_sequence_length - len(sentence)

            for i in range(words_to_pad):
                sentence.append('PADDING_TOKEN')

            # create data input for words
            for w_i, word in enumerate(sentence):

                if word == 'PADDING_TOKEN':
                    word_vector = [0] * nb_embedding_dims
                else:
                    word_vector = ft.get_word_vector(word.lower())
                temp_word.append(word_vector)

            word_embeddings.append(temp_word)
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(word_embeddings)], np.array(output_labels))

    
class TextFeatureGenerator2(Sequence):
    def __init__(self, text_data, batch_size = 32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):

        word_embeddings = []
        prev_embeddings = []
        output_labels = []

        for index in inds:
            
            word_embeddings.append(self.text_to_embedding(self.text_data[index][2]))
            prev_embeddings.append(self.text_to_embedding(self.text_data[index][3]))
            
            temp_output = label2Idx[self.text_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(word_embeddings), np.array(prev_embeddings)], np.array(output_labels))
    
    def text_to_embedding(self, textsequence):
        temp_word = []
            
        # tokenize
        sentence = simple_tokenizer(textsequence)
        
        # trim to max sequence length
        if (len(sentence) > nb_sequence_length):
            half_idx = int(nb_sequence_length / 2)
            tmp_sentence = sentence[:half_idx]
            tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
            sentence = tmp_sentence

        # padding
        words_to_pad = nb_sequence_length - len(sentence)

        for i in range(words_to_pad):
            sentence.append('PADDING_TOKEN')

        # create data input for words
        for w_i, word in enumerate(sentence):

            if word == 'PADDING_TOKEN':
                word_vector = [0] * nb_embedding_dims
            else:
                word_vector = ft.get_word_vector(word.lower())
            
            temp_word.append(word_vector)
            
        return temp_word    
    

def compile_model_singlepage(print_summary = False):
    model_input_ft = Input(shape = (nb_sequence_length, nb_embedding_dims))
    
    gru_block = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_ft)
    
    filter_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
        )(gru_block)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks.append(conv)
    model_concatenated = concatenate(conv_blocks)
    model_concatenated = Dense(128)(model_concatenated)
    model_concatenated = LeakyReLU()(model_concatenated)
    model_output = Dense(1, activation = "sigmoid")(model_concatenated)
    model = Model([model_input_ft], model_output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    if print_summary:
        model.summary()
    return model

def compile_model_prevpage(include_top = True, print_summary = False):
    
    filter_sizes = (3, 4, 5)
    
    model_input_tp = Input(shape = (nb_sequence_length, nb_embedding_dims))    
    gru_block_tp = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_tp)
    conv_blocks_tp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
        )(gru_block_tp)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks_tp.append(conv)
    model_concatenated_tp = concatenate(conv_blocks_tp)
    model_concatenated_tp = Dense(128)(model_concatenated_tp)
    model_concatenated_tp = LeakyReLU()(model_concatenated_tp)
    
    model_input_pp = Input(shape = (nb_sequence_length, nb_embedding_dims))    
    gru_block_pp = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_pp)
    conv_blocks_pp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
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
    
    if include_top:
        # prediction layer
        model_output = Dense(1, activation = "sigmoid")(page_sequence)

        # combine final model
        model = Model([model_input_tp, model_input_pp], model_output)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics = ['accuracy'])
        
        if print_summary:
            model.summary()
        
        return model
    else:
        model = Model([model_input_tp, model_input_pp], page_sequence)
        
        if print_summary:
            model.summary()
        
        return model_input_tp, model_input_pp, model
        



def predict(model, data, prev_page_generator = False, batch_size=256):
    if prev_page_generator:
        y_predict = np.round(model.predict_generator(TextFeatureGenerator2(data, batch_size=batch_size)))
    else:
        y_predict = np.round(model.predict_generator(TextFeatureGenerator(data, batch_size=batch_size)))
    return y_predict


class ValidationCheckpoint(Callback):
    def __init__(self, filepath, test_data, prev_page_generator = False, metric = 'kappa'):
        self.test_data = test_data
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        self.prev_page_generator = prev_page_generator

    def on_epoch_end(self, epoch, logs={}):
        
        predicted_labels = predict(self.model, self.test_data, self.prev_page_generator)
        true_labels = [label2Idx[x[1]] for x in self.test_data]

        eval_metrics = {
            'accuracy' : sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro' : sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro' : sklm.f1_score(true_labels, predicted_labels, average='macro'),
            'f1_binary' : sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label = 1),
            'kappa' : sklm.cohen_kappa_score(true_labels, predicted_labels)
        }
        eval_metric = eval_metrics[self.metric]
        self.history.append(eval_metric)
        
        if epoch > -1 and eval_metric > self.max_metric:
            print("\n" + self.metric + " improvement: " + str(eval_metric) + " (before: " + str(self.max_metric) + "), saving to " + self.filepath)
            self.max_metric = eval_metric     # optimization target
            self.max_metrics = eval_metrics   # all metrics
            self.model.save(self.filepath)

