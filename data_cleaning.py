import os, sys, random

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from skimage.transform import resize
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

import tensorflow as tf
from tensorflow.keras import models, Input

from sklearn.model_selection import train_test_split

# Set config
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = 'salt_image_data/train/'
    path_test = 'salt_image_data/test/'


# Plot sample images
random.seed(19)
ids = random.choices(os.listdir('salt_image_data/train/images'), k=6)
fig = plt.figure(figsize=(20, 6))
for j, img_name in enumerate(ids):
    q = j + 1

    img = load_img('salt_image_data/train/images/' + img_name)
    img_mask = load_img('salt_image_data/train/masks/' + img_name)

    plt.subplot(2, 6, q * 2 - 1)
    plt.imshow(img)
    plt.subplot(2, 6, q * 2)
    plt.imshow(img_mask)
fig.suptitle('Sample Images', fontsize=24)

# return list of filenames
train_ids = os.listdir('salt_image_data/train/images')  # Filenames in train/images
test_ids  = os.listdir('salt_image_data/test/images')   # Filenames in test/images

# print(train_ids)
# print(test_ids)


# Identify X and Y
X = np.zeros((len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool_)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    x = img_to_array(load_img(config.path_train + '/images/' + id_, color_mode="grayscale"))
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X[n] = x
    mask = img_to_array(load_img(config.path_train + '/masks/' + id_, color_mode="grayscale"))
    Y[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

# print('We get X and Y:')
# print('X shape:', X.shape) # (4000, 128, 128, 1)
# print('Y shape:', Y.shape) # (4000, 128, 128, 1)


# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # for reproducibility (optional)
    shuffle=True         # shuffles the data before splitting (default True)
)

X_train = np.append(X_train, [np.fliplr(x) for x in X], axis=0)
Y_train = np.append(Y_train, [np.fliplr(x) for x in Y], axis=0)
X_train = np.append(X_train, [np.flipud(x) for x in X], axis=0)
Y_train = np.append(Y_train, [np.flipud(x) for x in Y], axis=0)

del X, Y

print('X train shape:', X_train.shape, 'X eval shape:', X_test.shape) # (11200, 128, 128, 1), (800, 128, 128, 1)
print('Y train shape:', Y_train.shape, 'Y eval shape:', Y_test.shape) # (11200, 128, 128, 1), (800, 128, 128, 1)
