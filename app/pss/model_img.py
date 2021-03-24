import csv, re, math
import sklearn.metrics as sklm
import fasttext
import numpy as np

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Nadam, Adam, RMSprop
from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import *
from keras.callbacks import ModelCheckpoint, Callback

from PIL import Image

# import image_slicer
# n_slices = 6
# dummy_image_tiles = image_slicer.slice('15_1_0280.png', n_slices, save = False)

from pss import model as model_txt

nb_embedding_dims = 300 # ft.get_dimension()
nb_sequence_length = 150
word_vectors_ft = {}
label2Idx = {'FirstPage' : 1, 'NextPage' : 0}

img_dims = {'tobacco':299, 'other':299}
img_path_template = 'data/Tobacco800/images/%s.tif.small.png'
img_path_template = 'data/archive20k/images/%s.png'

def simple_tokenizer(textline):
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    # print(words)
    return(words)


def read_csv_data(csvfile, csvformat="Archive20k", return_DC = False):
    
    data_instances = []
    
    prevBinder = ""
    prevPageId = ""
    prevPageText = ""
    prevPageClass = ""
    
    with open(csvfile, 'r', encoding='UTF-8') as f:
        datareader = csv.reader(f, delimiter=';', quotechar='"')
        next(datareader)
        for instance in datareader:
            # "0 binder";"1 docid";"2 class";"3 type";"4 text"
            if prevBinder != instance[0]:
                prevPageId = ""
                prevPageText = ""
                
            if csvformat == "Tobacco800":
                data_instances.append([instance[1], instance[2], instance[3], prevPageId, prevPageText])
                prevBinder = instance[0]
                prevPageId = instance[1] 
                prevPageText = instance[3]
                prevPageClass = instance[2]
            else:
                if return_DC:
                    if instance[2] == "FirstPage" or prevPageClass == "FirstPage":
                        data_instances.append([instance[1], instance[2], instance[4], prevPageId, prevPageText, instance[3]])
                else:
                    data_instances.append([instance[1], instance[2], instance[4], prevPageId, prevPageText])
                prevBinder = instance[0]
                prevPageId = instance[1] 
                prevPageText = instance[4]
                prevPageClass = instance[2]
    return data_instances


class ImageFeatureGenerator(Sequence):
    def __init__(self, image_data, img_dim, prevpage = False, slices = False, batch_size = 32):
        self.image_data = image_data
        self.indices = np.arange(len(self.image_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []
        self.img_dim = img_dim
        self.prevpage = prevpage
        self.slices = slices

    def __len__(self):
        return math.ceil(len(self.image_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.prevpage:
            batch_x, batch_y = self.process_image_data_prevpage(inds)
        else:
            if self.slices:
                 batch_x, batch_y = self.process_image_slices(inds)
            else:
                batch_x, batch_y = self.process_image_data(inds)
        return batch_x, batch_y

    def process_image_data(self, inds):

        image_array = []
        output_labels = []

        for index in inds:
            image = self.get_image_data(self.image_data[index][0])
            image_array.append(image)
            
            temp_output = label2Idx[self.image_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(image_array)], np.array(output_labels))
    
    
    
    def process_image_slices(self, inds):

        image_array = []
        output_labels = []

        for index in inds:
            image = self.get_image_slices(self.image_data[index][0])
            image_array.append(image)
            
            temp_output = label2Idx[self.image_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)
            
        input_features = [np.array(l) for l in list(zip(*image_array))]
        
        return (input_features, np.array(output_labels))

    
    
    def process_image_data_prevpage(self, inds):

        image_array = []
        prev_image_array = []
        output_labels = []

        for index in inds:
            image = self.get_image_data(self.image_data[index][0])
            image_array.append(image)
            
            if self.image_data[index][3] != "":
                prev_image = self.get_image_data(self.image_data[index][3])
            else:
                prev_image = np.zeros((self.img_dim[0], self.img_dim[1], 3))
            prev_image_array.append(prev_image)
            
            temp_output = label2Idx[self.image_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(image_array), np.array(prev_image_array)], np.array(output_labels))
    
    def get_image_data(self, image_id):
        image_file = img_path_template % image_id 
        image = img_to_array(load_img(image_file, target_size=(self.img_dim[0], self.img_dim[1])))
        image = preprocess_input(image)
        return(image)
    
    def get_image_slices(self, image_id):
        image_file = img_path_template % image_id
        tiles = image_slicer.slice(image_file, n_slices, save = False)
        slices = []
        for tile in tiles:
            image = img_to_array(tile.image)
            image = np.repeat(image, 3, axis=2)
            slice = preprocess_input(image)
            slices.append(slice)
        return(slices)
        


class ImageFeatureGenerator2(Sequence):
    def __init__(self, image_data, img_dim, image_binary_array, prevpage = False, batch_size = 32):
        self.image_data = image_data
        self.indices = np.arange(len(self.image_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []
        self.img_dim = img_dim
        self.prevpage = prevpage
        self.image_binary_array = image_binary_array


    def __len__(self):
        return math.ceil(len(self.image_data) / self.batch_size)
    

    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)


    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.prevpage:
            batch_x, batch_y = self.process_image_data_prevpage(inds)
        else:
            batch_x, batch_y = self.process_image_data(inds)
        return batch_x, batch_y


    def process_image_data(self, inds):
        image_array = []
        output_labels = []

        for index in inds:
            image = self.get_image_data(self.image_data[index][0])
            image_array.append(image)
            
            temp_output = label2Idx[self.image_data[index][1]]
            output_labels.append(temp_output)

        return ([np.array(image_array)], np.array(output_labels))
    
    
    def process_image_data_prevpage(self, inds):

        image_array = []
        prev_image_array = []
        output_labels = []

        for index in inds:
            image = self.get_image_data(self.image_data[index][0])
            image_array.append(image)
            
            if self.image_data[index][3] != "":
                prev_image = self.get_image_data(self.image_data[index][3])
            else:
                prev_image = np.zeros((self.img_dim[0], self.img_dim[1], 3))
            prev_image_array.append(prev_image)
            
            temp_output = label2Idx[self.image_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(image_array), np.array(prev_image_array)], np.array(output_labels))
    

    def get_image_data(self, image_id):
        #image_file = img_path_template % image_id 
        #image = img_to_array(load_img(image_file, target_size=(self.img_dim[0], self.img_dim[1])))
        binary_image = self.image_binary_array[int(image_id)]
        #img = Image(binary_image)
        img = binary_image.resize((self.img_dim[0], self.img_dim[1]), Image.NEAREST)
        image = img_to_array(img)
        image = preprocess_input(image)
        return(image)
    


class ImageTextFeatureGenerator(Sequence):
    def __init__(self, image_data, img_dim, batch_size = 32):
        self.image_data = image_data
        self.indices = np.arange(len(self.image_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []
        self.img_dim = img_dim

    def __len__(self):
        return math.ceil(len(self.image_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_image_text_data(inds)
        return batch_x, batch_y


    def process_image_text_data(self, inds):

        image_array = []
        
        word_embeddings = []
        
        output_labels = []

        for index in inds:
            
            # image features
            image = self.get_image_data(self.image_data[index][0])
            image_array.append(image)
            
            # text features
            word_embeddings.append(self.text_to_embedding(self.image_data[index][2]))
            
            # output
            temp_output = doclabel2Idx[self.image_data[index][5]]
            output_labels.append(temp_output)

        output_labels = to_categorical(output_labels, len(idx2Doclabel)) # num_classes
        return ([np.array(image_array), np.array(word_embeddings)], output_labels)
    
    def get_image_data(self, image_id):
        image_file = img_path_template % image_id
        image = img_to_array(load_img(image_file, target_size=(self.img_dim[0], self.img_dim[1])))
        image = preprocess_input(image)
        return(image)
                
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
        



def compile_model_singlepage(img_dim, include_top = True, print_summary = False):
    
    model_vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))
    
    for l in model_vgg16.layers[:13]:
        l.trainable = False
    
    if print_summary:
        model_vgg16.summary()
        
    top_model = Flatten()(model_vgg16.output)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(512)(top_model)
    top_model = LeakyReLU()(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(256)(top_model)
    top_model = LeakyReLU()(top_model)
    if include_top:
        model_output = Dense(1, activation = "sigmoid")(top_model)
        model = Model(model_vgg16.input, model_output)
        model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.00001), metrics = ['accuracy'])
        if print_summary:
            model.summary()
        return model
    else:
        model = Model(model_vgg16.input, top_model)
        return model_vgg16.input, model
    

def compile_model_singlepage_slices(img_dim, print_summary = False):
    
    height = dummy_image_tiles[0].image.height
    width = dummy_image_tiles[0].image.width
    
    model_vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(height, width, 3))
    
    for l in model_vgg16.layers[:13]:
        l.trainable = False
    
    if print_summary:
        model_vgg16.summary()
        
    vgg_flat = Flatten()(model_vgg16.output)
    vgg_dropout = Dropout(0.5)(vgg_flat)

    top_model = Dense(512)(vgg_dropout)
    top_model = LeakyReLU()(top_model)
    top_model = Dropout(0.5)(top_model)
    
    shared_image_model = Model(model_vgg16.input, top_model)
    
    all_inputs = []
    all_features = []
    for i_slice in range(n_slices):
        slice_input = Input(shape = (height, width, 3))
        all_inputs.append(slice_input)
        tile_features = shared_image_model(slice_input)
        all_features.append(tile_features)
    
#     feat_rnn = concatenate(all_features)
#     feat_rnn = Reshape((n_slices, 512))(feat_rnn)
#     feat_rnn = Bidirectional(GRU(200))(feat_rnn)

    feat_slices = concatenate(all_features)
    feat_final = Dense(256)(feat_slices)
    feat_final = LeakyReLU()(feat_final)
    
    model_output = Dense(1, activation = "sigmoid")(feat_final)
    
    model = Model(all_inputs, model_output)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.00005), metrics = ['accuracy'])
    if print_summary:
        model.summary()
    return model



def compile_model_prevpage(img_dim, include_top = True, print_summary = False):
    # pp: prev_page
    # tp: target_page
    
    model_vgg16_pp = VGG16(weights = 'imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))
    model_vgg16_tp = VGG16(weights = 'imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))
    
    for i in range(13):
        model_vgg16_tp.layers[i].trainable = False
        model_vgg16_pp.layers[i].trainable = False
    for l in model_vgg16_pp.layers:
        l._name = 'pp_' + l.name
        # COMMENT l.name = 'pp_' + l.name
    
    vgg_flat_tp = Flatten()(model_vgg16_tp.output)
    vgg_dropout_tp = Dropout(0.5)(vgg_flat_tp)
    vgg_dense_tp = Dense(256)(vgg_dropout_tp)
    vgg_dense_tp = LeakyReLU()(vgg_dense_tp)
    
    vgg_flat_pp = Flatten()(model_vgg16_pp.output)
    vgg_dropout_pp = Dropout(0.5)(vgg_flat_pp)
    vgg_dense_pp = Dense(256)(vgg_dropout_pp)
    vgg_dense_pp = LeakyReLU()(vgg_dense_pp)

    # concat both + another dense
    page_sequence = concatenate([vgg_dense_tp, vgg_dense_pp])
    top_model = Dense(256)(page_sequence)
    top_model = LeakyReLU()(top_model)

    if include_top:
        # prediction layer
        model_output = Dense(1, activation = "sigmoid")(top_model)

        # combine final model
        model = Model([model_vgg16_tp.input, model_vgg16_pp.input], model_output)
        model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.00002), metrics = ['accuracy'])
        
        if print_summary:
            model.summary()
        return model

    else:
        model = Model([model_vgg16_tp.input, model_vgg16_pp.input], top_model)
        if print_summary:
            model.summary()
        return model_vgg16_tp.input, model_vgg16_pp.input, model


def compile_model_text_image(img_dim, n_classes, print_summary = False):
    img_input, image_model = compile_model_singlepage(img_dim, include_top=False)
    txt_input, text_model = model_txt.compile_model_singlepage(include_top=False)
    
    concat_model = concatenate([image_model.output, text_model.output])
    concat_model = Dense(256)(concat_model)
    concat_model = LeakyReLU()(concat_model)
    
    model_output = Dense(n_classes, activation = "softmax")(concat_model)
    
    model = Model([img_input, txt_input], model_output)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.00002), metrics = ['accuracy'])
    
    if print_summary:
        model.summary()
        
    return model
    
def predict(model, data, img_dim, prev_page_generator = False, batch_size=32):
    if prev_page_generator == "image+text":
        y_predict = model.predict_generator(ImageTextFeatureGenerator(data, img_dim, batch_size=batch_size)).argmax(axis=-1)
    elif prev_page_generator == "slices":
        y_predict = np.round(model.predict_generator(ImageFeatureGenerator(data, img_dim, slices=True, batch_size=batch_size)))
    elif prev_page_generator:
        y_predict = np.round(model.predict_generator(ImageFeatureGenerator(data, img_dim, prevpage=True, batch_size=batch_size)))
    else:
        y_predict = np.round(model.predict_generator(ImageFeatureGenerator(data, img_dim, prevpage=False, batch_size=batch_size)))
    
    return y_predict

class ValidationCheckpoint(Callback):
    def __init__(self, filepath, test_data, img_dim, prev_page_generator = False, metric = 'kappa'):
        self.test_data = test_data
        self.img_dim = img_dim
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        self.prev_page_generator = prev_page_generator

    def on_epoch_end(self, epoch, logs={}):
        
        predicted_labels = predict(self.model, self.test_data, self.img_dim, self.prev_page_generator)
        if self.prev_page_generator == "image+text":
            true_labels = [doclabel2Idx[x[5]] for x in self.test_data]
        else:
            true_labels = [label2Idx[x[1]] for x in self.test_data]

        eval_metrics = {
            'accuracy' : sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro' : sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro' : sklm.f1_score(true_labels, predicted_labels, average='macro'),
            # 'f1_binary' : sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label = 1),
            'kappa' : sklm.cohen_kappa_score(true_labels, predicted_labels)
        }
        eval_metric = eval_metrics[self.metric]
        self.history.append(eval_metric)
        
        if epoch > -1 and eval_metric > self.max_metric:
            print("\n" + self.metric + " improvement: " + str(eval_metric) + " (before: " + str(self.max_metric) + "), saving to " + self.filepath)
            self.max_metric = eval_metric     # optimization target
            self.max_metrics = eval_metrics   # all metrics
            self.model.save(self.filepath)

