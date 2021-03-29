import math
import csv
import numpy as np
import sklearn.metrics as sklm


from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, Callback
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Nadam
from keras.models import Model
from keras.layers import *
from keras.utils import *
from PIL import Image


LABEL2IDX = {'FirstPage' : 1, 'NextPage' : 0}
img_path_template = ""


def read_csv_data(csvfile):
    data_instances = []
    prev_image_file = ""

    # CSV Columns: "counter";"documentText";"label";"documentName"
    with open(csvfile, 'r', encoding='UTF-8') as f:
        datareader = csv.reader(f, delimiter=';', quotechar='"')
        next(datareader)
        for counter, csv_row in enumerate(datareader):
            data_instances.append([csv_row[0], csv_row[3], prev_image_file, csv_row[2]])
            prev_image_file = csv_row[3]
        return data_instances


class ImageFeatureGenerator(Sequence):
    def __init__(self, image_data, img_dim, prevpage=False, train=False, batch_size=32):
        self.image_data = image_data
        self.indices = np.arange(len(self.image_data))
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.prevpage = prevpage
        self.train = train

    def __len__(self):
        return math.ceil(len(self.image_data) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.prevpage:
            batch_x = self.process_image_data_prevpage(inds, file_mode=self.train)
        else:
            batch_x = self.process_image_data(inds, file_mode=self.train)

        if self.train:
            batch_y = self.generate_output_labels(inds)
        else:
            batch_y = np.zeros(int(len(inds)))
        return batch_x, batch_y

    def get_image_data(self, binary_image):
        img = binary_image.resize((self.img_dim[0], self.img_dim[1]), Image.NEAREST)
        image = img_to_array(img)
        image = preprocess_input(image)
        return image

    def get_image_data_from_file(self, image_id):
        image_file = img_path_template % image_id 
        image = img_to_array(load_img(image_file, target_size=(self.img_dim[0], self.img_dim[1])))
        image = preprocess_input(image)
        return image

    def process_image_data(self, inds, file_mode=False):
        image_array = []
        for index in inds:
            if file_mode:
                image = self.get_image_data_from_file(self.image_data[index][1])
            else:
                image = self.get_image_data(self.image_data[index][1])
            image_array.append(image)
        return [np.array(image_array)]

    def process_image_data_prevpage(self, inds, file_mode=False):
        image_array = []
        prev_image_array = []

        for index in inds:
            if file_mode:
                image = self.get_image_data_from_file(self.image_data[index][1])
            else:
                image = self.get_image_data(self.image_data[index][1])
            image_array.append(image)

            if self.image_data[index][2] != "":
                if file_mode:
                    prev_image = self.get_image_data_from_file(self.image_data[index][2])
                else:
                    prev_image = self.get_image_data(self.image_data[index][2])
            else:
                prev_image = np.zeros((self.img_dim[0], self.img_dim[1], 3))
            prev_image_array.append(prev_image)

        return [np.array(image_array), np.array(prev_image_array)]

    def generate_output_labels(self, inds):
        output_labels = []
        for index in inds:
            temp_output = LABEL2IDX[self.image_data[index][3]]
            output_labels.append(temp_output)
        return np.array(output_labels)


def predict(model, data, img_dim, prev_page_generator=False, batch_size=32):
    y_predict = np.round(model.predict(ImageFeatureGenerator(data, img_dim, prevpage=prev_page_generator, batch_size=batch_size)))
    return y_predict


def predict_without_rounding(model, data, img_dim, prev_page_generator=False, batch_size=32):
    y = model.predict(ImageFeatureGenerator(data, img_dim, prevpage=prev_page_generator, batch_size=batch_size))
    return y


def compile_model_singlepage(img_dim, print_summary=False):
    model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))

    for layer in model_vgg16.layers[:13]:
        layer.trainable = False

    if print_summary:
        model_vgg16.summary()

    top_model = Flatten()(model_vgg16.output)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(512)(top_model)
    top_model = LeakyReLU()(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(256)(top_model)
    top_model = LeakyReLU()(top_model)

    model_output = Dense(1, activation="sigmoid")(top_model)
    model = Model(model_vgg16.input, model_output)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.00001), metrics=['accuracy'])
    if print_summary:
        model.summary()
    return model


def compile_model_prevpage(img_dim, print_summary=False):
    model_vgg16_pp = VGG16(weights='imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))
    model_vgg16_tp = VGG16(weights='imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))

    for i in range(13):
        model_vgg16_tp.layers[i].trainable = False
        model_vgg16_pp.layers[i].trainable = False
    for layer in model_vgg16_pp.layers:
        layer._name = 'pp_' + layer.name

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

    # prediction layer
    model_output = Dense(1, activation="sigmoid")(top_model)

    # combine final model
    model = Model([model_vgg16_tp.input, model_vgg16_pp.input], model_output)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.00002), metrics=['accuracy'])

    if print_summary:
        model.summary()
    return model


class ValidationCheckpoint(Callback):
    def __init__(self, filepath, test_data, img_dim, prev_page_generator=False, metric='kappa'):
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
        true_labels = [LABEL2IDX[x[1]] for x in self.test_data]

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
