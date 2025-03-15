import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers

import data_cleaning

# Normalize x so that apply activation function ReLU
def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# Create a convolution kernel
def convolution_layers(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

#residual connects to the stacking blocks
def residual_connect(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_layers(x, num_filters, (3,3)) # 3x3 convolution block
    x = convolution_layers(x, num_filters, (3,3), activation=False)
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # apply a smaller scale
    scaled = layers.Lambda(lambda x: x / 255)(input_layer)

    # encoding
    # from size 101x101 to 50x50
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(scaled)
    conv1 = residual_connect(conv1, start_neurons * 1)
    conv1 = residual_connect(conv1, start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    # from size 50x50 to 25x25
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_connect(conv2, start_neurons * 2)
    conv2 = residual_connect(conv2, start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    # from size 25x25 to 12x12
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_connect(conv3, start_neurons * 4)
    conv3 = residual_connect(conv3, start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    # from size 12x12 to 6x6
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_connect(conv4, start_neurons * 8)
    conv4 = residual_connect(conv4, start_neurons * 8, True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(DropoutRatio)(pool4)

    # deepest layer, decoding
    conv = layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    conv = residual_connect(conv, start_neurons * 16)
    conv = residual_connect(conv, start_neurons * 16, True)

    # from size 6x6 to 12x12
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(conv)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DropoutRatio)(uconv4)

    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_connect(uconv4, start_neurons * 8)
    uconv4 = residual_connect(uconv4, start_neurons * 8, True)

    # from size 12x12 to 25x25
    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_connect(uconv3, start_neurons * 4)
    uconv3 = residual_connect(uconv3, start_neurons * 4, True)

    # from size 25x25 to 50x50
    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_connect(uconv2, start_neurons * 2)
    uconv2 = residual_connect(uconv2, start_neurons * 2, True)

    # from size 50x50 to 101x101
    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_connect(uconv1, start_neurons * 1)
    uconv1 = residual_connect(uconv1, start_neurons * 1, True)

    # uconv1 = layers.Dropout(DropoutRatio/2)(uconv1)
    # output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = layers.Activation('sigmoid')(output_layer_noActi)

    return output_layer


input_layer = Input((data_cleaning.config.im_height,
                     data_cleaning.config.im_width,
                     data_cleaning.config.im_chan)) # (height, width, channels)
output_layer = build_model(input_layer, 16)

model = models.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # accuracy
model.summary() # Trainable params: 5,112,497, Non-trainable params: 7,360

# plot model
# utils.plot_model(model, expand_nested=True, show_shapes=True)

es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    data_cleaning.X_train, data_cleaning.Y_train, validation_data=(data_cleaning.X_test,
                                                                   data_cleaning.Y_test),
                                                                   batch_size=8,
                                                                   epochs=300,
                                                                   callbacks=[es, rlp]
)

sns.set_style('darkgrid')
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[['loss', 'val_loss']].plot(ax=ax[0])
history[['acc', 'val_acc']].plot(ax=ax[1])
fig.suptitle('Learning Curve', fontsize=24)
