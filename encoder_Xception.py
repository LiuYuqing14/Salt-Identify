import numpy as np

from skimage.transform import resize
import model_construct
import data_cleaning

from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
import keras.callbacks as callbacks
from keras.applications.xception import Xception



img_size_ori = 101 # 101 -> 128
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if str(rle_mask)==str(np.nan):
        return np.zeros((101,101))
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)


def UXception(input_shape=(None, None, 3)):
    backbone = Xception(input_shape=input_shape, weights='imagenet', include_top=False)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = model_construct.residual_connect(convm, start_neurons * 32)
    convm = model_construct.residual_connect(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    # 10 -> 20
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = model_construct.residual_connect(uconv4, start_neurons * 16)
    uconv4 = model_construct.residual_connect(uconv4, start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    # 10 -> 20
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.1)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = model_construct.residual_connect(uconv3, start_neurons * 8)
    uconv3 = model_construct.residual_connect(uconv3, start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 20 -> 40
    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1, 0), (1, 0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = model_construct.residual_connect(uconv2, start_neurons * 4)
    uconv2 = model_construct.residual_connect(uconv2, start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    # 40 -> 80
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3, 0), (3, 0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = model_construct.residual_connect(uconv1, start_neurons * 2)
    uconv1 = model_construct.residual_connect(uconv1, start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    # 80 -> 160
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = model_construct.residual_connect(uconv0, start_neurons * 1)
    uconv0 = model_construct.residual_connect(uconv0, start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(0.1 / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model

K.clear_session()
model = UXception(input_shape=(img_size_target,img_size_target,3))

class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("./keras.model",monitor='val_my_iou_metric',
                                   mode = 'max', save_best_only=True, verbose=1),
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

epochs = 40
snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
batch_size = 32
history = model.fit(data_cleaning.X_train, data_cleaning.Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=snapshot.get_callbacks(),shuffle=True,verbose=2)