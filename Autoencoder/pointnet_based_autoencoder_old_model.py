import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
import random
import time
import sys

from sklearn.neighbors import KDTree
from tensorflow.keras import backend as K
K.clear_session()

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(1234)



#directory = '/content/drive/My Drive/Data_Lidar/Data/data_32.zip (Unzipped Files)/data_32'
directory = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\training folder\\data_32'
#directory = 'C:\\Users\\ElHaddar\\OneDrive\\Desktop\\training folder\\data_32\\file'

list_pcl = []
for filename in os.listdir(directory):
    if filename.endswith(".ply") :
        directory_file = os.path.join(directory, filename)
        print(directory_file)
        pcl = PlyData.read(directory_file)
        data = pcl.elements[0].data
        data = np.asarray(data.tolist())
        #data.resize(1024,3)
        M = np.abs(1024 - data.shape[0])
        a = np.pad(data, ((0,M),(0,0)), 'symmetric')
        data.resize(1024,3)
        #print(type(data))
        #print(data.shape)
        list_pcl.append(data)
print(len(list_pcl))


X = np.asarray(list_pcl[0:]).astype("float32")
X_val = np.asarray(list_pcl[74000:74299]).astype("float32")
X_test = np.asarray(list_pcl[74300:74329]).astype("float32")

#X = np.asarray(list_pcl[0:150]).astype("float32")
#X_val = np.asarray(list_pcl[151:200]).astype("float32")
#X_test = np.asarray(list_pcl[201:207]).astype("float32")

random.shuffle(X)
random.shuffle(X_val)
random.shuffle(X_test)

"""**Reshaping the dataset**

The neural network is unable to treat data with different input size, that's why we apply a zero padding to all the data to reach the size of the point cloud data with the biggest number of raws. 
We additioally reshape the outcome by adding one dimension corresponidng to the number of channels to the tesors.
"""

train_num = X.shape[0]
val_num = X_val.shape[0]
test_num = X_test.shape[0]
points_num = X.shape[1]
features_num = X.shape[2]

train_data = X.reshape([-1, points_num, features_num])
val_data = X_val.reshape([-1, points_num, features_num])
test_data = X_test.reshape([-1, points_num, features_num])

print(train_data.shape)
print(type(train_data))
print(val_data.shape)
print(test_data.shape)


"""### **Build a model**


Each convolution and fully-connected layer (with exception for end layers) consits of Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

"""PointNet consists of two core components. The primary MLP network, and the transformer net (T-net). The T-net aims to learn an affine transformation matrix by its own mini network. The T-net is used twice. The first time to transform the input features (n, 3) into a canonical representation. The second is an affine transformation for alignment in feature space (n, 3). As per the original paper we constrain the transformation to be close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0)."""

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

""" We can then define a general function to build T-net layers.

"""

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

"""The main network can be then implemented in the same manner where the t-net mini models can be dropped in a layers in the graph. 

**Building the Encoder**
"""

inputs = keras.Input(shape=(1024, 3))

#x = tnet(inputs, 3)
x = conv_bn(inputs, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
#x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = conv_bn(x, 1024)
#x = layers.Conv1D(32, kernel_size=1, padding="valid")(inputs)
#x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(64, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(512, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(1024, kernel_size=1, padding="valid")(x)
x = layers.Conv1D(1536, kernel_size=1, padding="valid")(x)
#x = conv_bn(x, 1536)

encoded = layers.GlobalMaxPooling1D()(x)

print("shape of encoded: ")
print(K.int_shape(encoded))

"""**Building the Decoder**"""
x = layers.Reshape((512, 3, 1))(encoded)
#x = layers.RepeatVector(2048)(encoded)

#x = tf.transpose(x, perm=[0, 2, 1])

#x = layers.Reshape((1024, 2048, 1))(x)
#x = conv_bn(x, 1024)
x = layers.Conv2D(1024, kernel_size=1)(x)
x = layers.UpSampling2D((2,1))(x)

x = layers.Conv2D(512, kernel_size=1)(x)
x = layers.Conv2D(64, kernel_size=1)(x)
x = layers.Conv2D(1, kernel_size=1)(x)
#x = dense_bn(encoded, 1536)
#x = dense_bn(x, 3072)
#x = layers.Dense(1024)(encoded)
#x = layers.Dense(1024)(x)
#x = layers.Dense(1024*3)(x)

#x = layers.Reshape((512, 3))(x)
#decoded = layers.UpSampling1D(2)(x)

#x = layers.Conv1D(1024, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(512, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(64, kernel_size=1, padding="valid")(x)
#x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)

#decoded = layers.Conv1D(3, kernel_size=1, padding="valid")(x)


decoded = layers.Reshape((points_num, features_num))(x)


print("shape of decoded: ")
print(K.int_shape(decoded))

"""**Model summary**"""

autoencoder = keras.Model(inputs=inputs, outputs=decoded, name="pointnet")
autoencoder.summary()


"""### **Create Loss Function : Chamfer distance**"""




def distance_matrix(array1, array2):

    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances

def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2

def chamfer_distance_tf(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = tf.reduce_mean(
               tf.map_fn(av_dist_sum, elems=(array1, array2), fn_output_signature=tf.float32))

    #print(type(dist))
    #print(dist.shape)

    return dist



#print(chamfer_distance_tf(X,X))


#Adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
autoencoder.compile(loss= chamfer_distance_tf, optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))

history = autoencoder.fit(train_data, train_data, epochs=10, batch_size=16, validation_data=(val_data, val_data))

autoencoder.save(r'C:\Users\ElHaddar\OneDrive\Desktop\Project\PointNet_based_autoencoder_32_int8_5')

print(history.history.keys())



# "Loss"

plt.plot(history.history['loss'])
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()






