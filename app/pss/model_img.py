import math
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Nadam
from keras.models import Model
from keras.layers import *
from keras.utils import *
from PIL import Image


class ImageFeatureGenerator(Sequence):
    def __init__(self, image_data, img_dim, prevpage=False, batch_size=32):
        self.image_data = image_data
        self.indices = np.arange(len(self.image_data))
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.prevpage = prevpage

    def __len__(self):
        return math.ceil(len(self.image_data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.prevpage:
            batch_x, batch_y = self.process_image_data_prevpage(inds)
        else:
            batch_x, batch_y = self.process_image_data(inds)
        return batch_x, batch_y

    def process_image_data(self, inds):
        image_array = []
        for index in inds:
            image = self.get_image_data(self.image_data[index][1])
            image_array.append(image)
        return [np.array(image_array)], np.zeros(int(len(inds)))

    def process_image_data_prevpage(self, inds):
        image_array = []
        prev_image_array = []

        for index in inds:
            image = self.get_image_data(self.image_data[index][1])
            image_array.append(image)

            if self.image_data[index][2] != "":
                prev_image = self.get_image_data(self.image_data[index][2])
            else:
                prev_image = np.zeros((self.img_dim[0], self.img_dim[1], 3))
            prev_image_array.append(prev_image)

        return [np.array(image_array), np.array(prev_image_array)], np.zeros(int(len(inds)))

    def get_image_data(self, binary_image):
        img = binary_image.resize((self.img_dim[0], self.img_dim[1]), Image.NEAREST)
        image = img_to_array(img)
        image = preprocess_input(image)
        return image


def predict(model, data, img_dim, prev_page_generator=False, batch_size=32):
    if prev_page_generator:
        y_predict = np.round(model.predict(ImageFeatureGenerator(data, img_dim, prevpage=True,
                                                                           batch_size=batch_size)))
    else:
        y_predict = np.round(model.predict(ImageFeatureGenerator(data, img_dim, prevpage=False,
                                                                           batch_size=batch_size)))
    return y_predict


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
