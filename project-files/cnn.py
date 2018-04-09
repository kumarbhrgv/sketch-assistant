import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from skimage.io import imread
from scipy import misc
import argparse
import os
from PIL import Image
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

image_dir1 = os.getcwd() + "/dataset/left_!/path_png_resize/"
x_train= []
y_train = []
for f in os.listdir(image_dir1):
    filename = os.fsdecode(f)
    print(filename)
    image = imread(image_dir1+filename)
    image = image/255
    image = 1-image
    #image = np.reshape(image,(image.shape[0]*image.shape[1]))
    x_train.append(image)
    y_train.append(1)
if False:
    pass
else:
    image_dir2 = os.getcwd() + "/sketches_png/png/dog/"
    for f in os.listdir(image_dir2):
        filename = os.fsdecode(f)
        print(filename)
        image = imread(image_dir2+"/"+filename)
        image = image/255
        image = 1-image
        #image = np.reshape(image,(image.shape[0],image.shape[1]))
        x_train.append(image)
        y_train.append(0)
x_train = np.reshape(x_train,(len(x_train),image.shape[0], image.shape[1],1))
print(x_train.shape)
from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y_train)
print(y_binary.shape)
input_shape = (image.shape[0], image.shape[1],1)
model = Sequential()
model.add(Conv2D(5, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,name='Conv1'))
model.add(Conv2D(20, (3, 3), activation='relu',name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(y_binary.shape[1], activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_binary,
          epochs=5,
          verbose=1)

from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'Conv2')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx])
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')