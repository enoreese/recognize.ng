# from .inception_blocks_v2 import *
from api.core import logger
import tensorflow as tf
import numpy as np
import os, math
import urllib.request
import cv2, base64
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import Model
from numpy import genfromtxt
from api.repositiory import FaceRepository, EmbeddingRepository
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
import h5py
from sklearn import neighbors
import pickle

_FLOATX = 'float32'
graph = None


def inception_block_1a(X):
    """
    Implementation of an inception block
    """

    X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name='inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(16, (1, 1), data_format='channels_first', name='inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)
    X_5x5 = Conv2D(32, (5, 5), data_format='channels_first', name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)), data_format='channels_first')(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    # CONCAT
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception


def inception_block_1b(X):
    X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name='inception_3b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3b_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)
    X_5x5 = Conv2D(64, (5, 5), data_format='channels_first', name='inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=(4, 4), data_format='channels_first')(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception


def inception_block_1c(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_3c_3x3',
                      cv1_out=128,
                      cv1_filter=(1, 1),
                      cv2_out=256,
                      cv2_filter=(3, 3),
                      cv2_strides=(2, 2),
                      padding=(1, 1))

    X_5x5 = conv2d_bn(X,
                      layer='inception_3c_5x5',
                      cv1_out=32,
                      cv1_filter=(1, 1),
                      cv2_out=64,
                      cv2_filter=(5, 5),
                      cv2_strides=(2, 2),
                      padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

    return inception


def inception_block_2a(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_4a_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=192,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                      layer='inception_4a_5x5',
                      cv1_out=32,
                      cv1_filter=(1, 1),
                      cv2_out=64,
                      cv2_filter=(5, 5),
                      cv2_strides=(1, 1),
                      padding=(2, 2))

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_4a_pool',
                       cv1_out=128,
                       cv1_filter=(1, 1),
                       padding=(2, 2))
    X_1x1 = conv2d_bn(X,
                      layer='inception_4a_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

    return inception


def inception_block_2b(X):
    # inception4e
    X_3x3 = conv2d_bn(X,
                      layer='inception_4e_3x3',
                      cv1_out=160,
                      cv1_filter=(1, 1),
                      cv2_out=256,
                      cv2_filter=(3, 3),
                      cv2_strides=(2, 2),
                      padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                      layer='inception_4e_5x5',
                      cv1_out=64,
                      cv1_filter=(1, 1),
                      cv2_out=128,
                      cv2_filter=(5, 5),
                      cv2_strides=(2, 2),
                      padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(X_pool)

    inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

    return inception


def inception_block_3a(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_5a_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=384,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_5a_pool',
                       cv1_out=96,
                       cv1_filter=(1, 1),
                       padding=(1, 1))
    X_1x1 = conv2d_bn(X,
                      layer='inception_5a_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))

    inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

    return inception


def inception_block_3b(X):
    X_3x3 = conv2d_bn(X,
                      layer='inception_5b_3x3',
                      cv1_out=96,
                      cv1_filter=(1, 1),
                      cv2_out=384,
                      cv2_filter=(3, 3),
                      cv2_strides=(1, 1),
                      padding=(1, 1))
    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                       layer='inception_5b_pool',
                       cv1_out=96,
                       cv1_filter=(1, 1))
    X_pool = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_pool)

    X_1x1 = conv2d_bn(X,
                      layer='inception_5b_1x1',
                      cv1_out=256,
                      cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

    return inception


def faceRecoModel(input_shape):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=1, name='bn1')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    # Second Block
    X = Conv2D(64, (1, 1), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size=3, strides=2)(X)

    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)

    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)

    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)

    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)

    # L2 normalization
    X = Lambda(lambda x: tf.nn.l2_normalize(x, dim=1))(X)

    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='FaceRecoModel')

    return model


def variable(value, dtype=_FLOATX, name=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    _get_session().run(v.initializer)
    return v


def shape(x):
    return x.get_shape()


def square(x):
    return tf.square(x)


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def __concatenate(tensors, axis=-1):
    if axis < 0:
        axis = axis % len(tensors[0].get_shape())
    return tf.concat(axis, tensors)


def LRN2D(x):
    return tf.nn.lrn(x, alpha=1e-4, beta=0.75)


def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer + '_conv' + num)(
        x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer + '_bn' + num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer + '_conv' + '2')(
        tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer + '_bn' + '2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor


WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
    'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'dense_layer'
]

conv_shape = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1],
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1, ],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1],
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1],
    'inception_5b_1x1_conv': [256, 736, 1, 1],
}


def load_weights_from_FaceNet(FRmodel):
    # Load weights from csv files (which was exported from Openface torch model)
    weights = WEIGHTS
    weights_dict = load_weights()

    # Set layer weights of the model
    for name in weights:
        if FRmodel.get_layer(name) != None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])
        elif model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])


def load_weights():
    # Set weights path
    dirPath = '/Users/sasu/Desktop/Dev/RecognizeNg/api/utils/weights/'
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath + '/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath + '/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url=url, timeout=20)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # from skimage import io
    # image = io.imread(url)

    # return the image
    return image


def img_to_encoding(url, model, resize=False, crop=True):
    print(url)
    print('inside')
    img1 = url_to_image(url)  # cv2.imread(image_file, 1)
    print('img to url')
    if crop:
        print('crop')
        detected, faces, message, eyes = detect_face(url=img1, skip=True)
        print(faces)
        if not detected:
            return False, "No face has been detected"
        if len(faces) > 1:
            return False, 'Multiple faces found'
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg =  img1[y:y + h, x:x + w]
            # lastimg = cv2.resize(faceimg, (96, 96))

    if resize:
        img1 = cv2.resize(faceimg, (96, 96))

    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    # with graph.as_default():
    embedding = model.predict_on_batch(x_train)
    return True, embedding


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by the formula

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = (pos_dist - neg_dist) + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


# def url_to_image(url):
#     # download the image, convert it to a NumPy array, and then read
#     # it into OpenCV format
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     # from skimage import io
#     # image = io.imread(url)
#
#     # return the image
#     return image


def verify(image_path, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    ### START CODE HERE ###

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    status, encoding = img_to_encoding(image_path, model, resize=True)
    if not status:
        return None, None, encoding

    dist = 0

    # Step 2: Compute distance with identity's image (≈ 1 line)
    for (name, db_enc) in database.items():

        dist += np.linalg.norm(db_enc - encoding)

    final_dist = dist / len(database)

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if final_dist < 0.7:
        print("welcome home!")
        match = True
    else:
        print("please go away")
        match = False

    ### END CODE HERE ###

    return final_dist, match, encoding

def gray_image(image_file):
    # imgtest = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    # imgtest = cv2.equalizeHist(imgtest)
    imgtest = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

    return imgtest

def detect_eyes(roi_gray):
    eye_cascade = cv2.CascadeClassifier('/Users/sasu/Desktop/Dev/RecognizeNg/api/utils/haar_classifiers/eye_detect_model.xml')

    eyes = eye_cascade.detectMultiScale(roi_gray)

    return eyes


def detect_face(url, skip=False):
    decoded_image = url
    if not skip:
        decoded_image = url_to_image(url=url)
    imgtest = gray_image(image_file=decoded_image)
    logger.info('cascade')
    face_cascade = cv2.CascadeClassifier('/Users/sasu/Desktop/Dev/RecognizeNg/api/utils/haar_classifiers/face_detect_model.xml')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("detect face")
    faces = face_cascade.detectMultiScale(imgtest, 1.1, 3, minSize=(100, 100))
    if (faces is None):
        print('Failed to detect face')
        return 0
    eyes = None
    cropped_image = None
    no_faces = len(faces)
    print("Detected faces: %d" % no_faces)

    if no_faces == 0:
        detected = False
        message = "No face has been detected"
    else:
        if no_faces == 1:
            detected = True
            message = "A face has been detected"
        else:
            detected = True
            message = "Multiple faces detected"

        eyes = []
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = imgtest[ny:ny + nr, nx:nx + nr]
            lastimg = cv2.resize(faceimg, (96, 96))
            cropped_image = lastimg

            eyes_ = detect_eyes(roi_gray=lastimg)

            for eye in eyes_:
                eyes.append(eye.tolist())

    return detected, faces, message, eyes


# def similarity_search(database, query):
#     d = query.shape[1]
#     index = faiss.IndexFlatL2(d)
#     print(index.is_trained)
#     index.add()


async def train_knn(user_id, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    print("Training KNN...")
    X = np.array([])# np.empty((0,128), float)
    Y = np.array([])# np.empty((0,1), float)

    faces = FaceRepository.getFaces(user_id=user_id)

    for face in faces:
        for embed in face.embeddings:
            embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=embed.id)
            # np.append(Y, np.array(int(user_id)), axis=0)
            # np.append(X, np.array(embedding.embedding).astype('float32'), axis=0)
            X = np.vstack((X, np.array(embedding.embedding).astype('float32')))
            Y = np.vstack((Y, np.array(int(user_id)).astype('int32')))
            # Y.append(int(user_id))
            # X.append(np.array(embedding.embedding).astype('float32'))

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    X = np.array(X)

    Y = np.array(Y)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, Y)

    # Save the trained KNN classifier
    model_save_path = 'api/utils/knn_classifiers/user_{}_trained_knn.clf'.format(user_id)
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    return knn_clf


def predict_face(encoding, user_id, knn_clf=None, distance_threshold=0.6):
    # Load a trained KNN model (if one was passed in)
    model_save_path = 'api/utils/knn_classifiers/user_{}_trained_knn.clf'.format(user_id)
    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(encoding, n_neighbors=1)
    print("closerst Distance", closest_distances)
    is_match = closest_distances[0][0][0] <= distance_threshold
    pred = knn_clf.predict(encoding)
    print("pred: ", pred)

    if is_match:
        message = 'Face found'
    else:
        message = 'No face found'

    return closest_distances[0][0][0], pred, message


def who_is_it_bulk(image_path, model, user_id, distance_threshold=0.6):
    ## Step 1: Compute the target "encoding" for the image.
    status, encoding = img_to_encoding(image_path, model, resize=True)
    if not status:
        return None, None, encoding, None

    ## Step 2: Find the closest encoding ##

    distance, identity, message = predict_face(encoding=encoding, user_id=user_id)

    return distance, identity, message, encoding


def who_is_it(image_path, keys, values, model):
    """
    Implements face recognition function for finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image.
    status, encoding = img_to_encoding(image_path, model, resize=True)
    if not status:
        return None, None, encoding, None

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in zip(keys, values):

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.76:
        print("Not in the database.")
        message = 'No face found'
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
        message = "Face found"

    return min_dist, identity, message, encoding


def create_model():
    global graph
    print('Building CNN Architecture...')
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))

    print("Total Params:", FRmodel.count_params())

    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    FRmodel._make_predict_function()

    return FRmodel