import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation

from skimage.io import imread
import os
import numpy as np
import pandas

image_dir1 = os.getcwd() + "/left_!/path_png_resize_updated/"
images = []
images_attr = []
df = pandas.read_csv(os.getcwd() + "/path_annotations_updated.csv")
df = df.sample(frac=1).reset_index(drop=True)
df_filenames = df['file']
df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop("Unnamed: 0.1", axis=1, inplace=True)
print(df.isnull().any())
count = 1
for filename in df_filenames.values:
    image = imread(image_dir1 + filename)
    image = image / 255
    image = image.reshape((200, 200, 1))
    images.append(image)
    images_attr.append(np.array((df.iloc[df.index[df["file"] == filename].tolist()].values[0])[1:-1]))
del df
import gc
gc.collect()

images = np.array(images)
images_attr = np.array(images_attr)
x_train = images[:1000]
y_train = images_attr[:1000]
x_test = images[1000:]
y_test = images_attr[1000:]
input_shape = (image.shape[0], image.shape[1],1)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train,
          epochs=10,
          verbose=1,batch_size= 20,validation_split=0.1)
score = model.evaluate(x_train[:20], y_train[:20], batch_size=10)
print(score)
