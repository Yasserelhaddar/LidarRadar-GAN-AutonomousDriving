# -*- coding: utf-8 -*-
"""Convolutional Autoencoder on Lidar scans.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cbaqPjW7t1mbs6-_BYaISyRSuoCsKBvW

Connect google colab notebook with google drive
"""


"""Access the drive and import compressed train, validation and test data."""

import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from time import time
from keras.layers import Input, Cropping3D, Dense, Flatten, Reshape
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D 


import matplotlib.pyplot as plt


from keras import backend as K
K.clear_session()


data1 = np.load('C:/Users/ElHaddar/OneDrive/Desktop/training folder/Real_data_4096.npz')
data2 = np.load('C:/Users/ElHaddar/OneDrive/Desktop/training folder/Sim_data_4096.npz')

X1 = shuffle(data1['a'])
X2 = shuffle(data2['a'])

X = np.concatenate((X1, X2), axis=0)
#X_validation = shuffle(data['X_validation'])
#X_test = shuffle(data['X_test'])

"""Show the size of train, test and validation data"""

print(X.shape)
#print(X_test.shape)
#print(X_validation.shape)

"""Reshaping the data """

train_num = X.shape[0]
#val_num = X_validation.shape[0]
#test_num = X_test.shape[0]
box_size = X.shape[1]


train_data = X.reshape([-1, box_size, box_size, box_size, 1])
#val_data = X_validation.reshape([-1, box_size, box_size, box_size, 1])
#test_data = X_test.reshape([-1, box_size, box_size, box_size, 1])

print(train_data.shape)
#print(val_data.shape)
#print(test_data.shape)


input_img = Input(shape=(64, 64, 64, 1))

x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same',name='Convolutional_Layer_1')(input_img)
x = MaxPooling3D((2, 2, 2), padding='same',name='MaxPooling_layer_1')(x)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same',name='Convolutional_Layer_2')(x)
x = MaxPooling3D((2, 2, 2), padding='same',name='MaxPooling_layer_2')(x)
x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same',name='Convolutional_Layer_3')(x)

encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)


print("shape of encoded: ")
print(K.int_shape(encoded))

"""Building the decoder"""

x = UpSampling3D((2, 2, 2))(encoded)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)

decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)

print("shape of decoded: ")
print(K.int_shape(decoded))

"""Creating the autoencoder model, compile it with adadelta optimizer and start the training process with 200 epochs and 100 batch_size"""

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

autoencoder.summary()


history = autoencoder.fit(train_data, train_data, epochs=30, batch_size=1)

autoencoder.save(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\VoxelNet_based_autoencoder')

print(history.history.keys())

#  "Accuracy"

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# "Loss"

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

