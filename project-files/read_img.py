import math
import numpy as np


def combine_normalized_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img
    return image


def img_from_normalized_img(normalized_img):
    image = normalized_img * 127.5 + 127.5
    return Image.fromarray(image.astype(np.uint8))



from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from PIL import Image
import os



class DCGan(object):
    model_name = 'dc-gan'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None

    def create_model(self):
        init_img_width = 200 // 4
        init_img_height = 200 // 4

        random_input = Input(shape=(9,))
        text_input1 = Input(shape=(9,))
        random_dense = Dense(1024)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(64 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height, 64),
                                  input_shape=(64 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(4, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, text_input1], generator_output)

        self.generator.compile(loss='mean_squared_error', optimizer="SGD")

        print('generator: ', self.generator.summary())

        text_input2 = Input(shape=(9,))
        text_layer2 = Dense(200)(text_input2)

        img_input2 = Input(shape=(200, 200,4))
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(
            img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(200)(img_layer2)

        merged = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        self.discriminator = Model([img_input2, text_input2], discriminator_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        print('discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, text_input1])

        self.model = Model([random_input, text_input1], model_output)
        self.discriminator.trainable = False

        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        print('generator-discriminator: ', self.model.summary())

    def load_model(self, model_dir_path):
        self.create_model()

    def fit(self, x_train,y_train, epochs=None, batch_size=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        noise = np.zeros((batch_size, 9))
        text_batch = np.zeros((batch_size, 9))
        self.create_model()
        for epoch in range(epochs):
            print("Epoch is", epoch)
            batch_count = int(len(x_train)// batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator

                x_train_batch = x_train[batch_index * batch_size:(batch_index + 1) * batch_size]
                y_train_batch = y_train[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image = x_train_batch[index]
                    image_batch.append(image)
                    text_batch[index, :] = y_train_batch[index]
                    noise[index, :] = np.random.uniform(-1, 1, 9)
                image_batch = np.array(image_batch)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                self.save_snapshots(generated_images,
                                        epoch=epoch, batch_index=batch_index)
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)),
                                                            np.concatenate((text_batch, text_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))
                print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))

                # Step 2: train the generator
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, 9)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))

    def generate_image_from_text(self, text):
        noise = np.zeros(shape=(1,9))
        generated_images = self.generator.predict([noise, text], verbose=0)
        generated_image = generated_images[0]
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_snapshots(self, generated_images, epoch, batch_index, snapshot_dir_path="./left_!/"):
        image = combine_normalized_images(generated_images)
        img_from_normalized_img(image).save(
            os.path.join(snapshot_dir_path, DCGan.model_name + '-' + str(epoch) + "-" + str(batch_index) + ".png"))


def average(pixel):
    x = 0.299*pixel[1] + 0.587*pixel[2] + 0.114*pixel[3]
    return x / 3


if __name__ == "__main__":
    from skimage.io import imread
    import os
    import numpy as np
    import pandas
    import math

    image_dir1 = os.getcwd() + "/left_!/path_png_resize/"
    images = []
    images_attr = []
    df = pandas.read_csv(os.getcwd() + "/path_annotations_updated.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    df_filenames = df['file']
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.drop("Unnamed: 0.1", axis=1, inplace=True)
    count = 1
    for filename in df_filenames.values:
        print(filename)
        image = imread(image_dir1 + filename)
        image = image / 255
        images.append(image)
        images_attr.append(np.array((df.iloc[df.index[df["file"] == filename].tolist()].values[0])[1:]))
        count +=1
        if count == 5:
            break
    del df
    import gc
    gc.collect()
    x_train = images[:500]
    y_train = images_attr[:500]
    x_test = images[:100]
    y_test = images_attr[:100]
    print("Train : " ,len(x_train))
    print("Train : ", len(x_test))
    model = DCGan()
    model.fit(x_train,y_train,10,1)
    model.generate_image_from_text(np.array(y_train[0]).reshape(1,9)).show()