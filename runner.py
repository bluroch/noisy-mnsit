# fmt: off
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
import numpy
import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
# fmt: on

# Global variables
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28

# Terminal flags for ease of use
verbose = False
debug = False
for arg in sys.argv:
    if arg == 'verbose':
        verbose = True
    if arg == 'debug':
        debug = True

# Base model to reference


def gaussian_noise(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(3, 3),
                     padding="same",
                     input_shape=(img_rows, img_cols, 1)))
    model.add(GaussianNoise(0.1))
    model.add(Activation(keras.activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    if (verbose):
        model.summary()

    return model.evaluate(x_test, y_test, verbose=0)


def base_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(3, 3),
                     padding="same",
                     input_shape=(img_rows, img_cols, 1)))
    model.add(Activation(keras.activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    return model.evaluate(x_test, y_test, verbose=0)

# Func list to run, add new funcs as they are made


def func_list():
    # funcs = [base_model, gaussian_noise]
    funcs = [gaussian_noise]
    return funcs

# Main functions


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if (debug):
        print('elements reduced from: ',  x_train.shape)
        x_train = numpy.split(x_train, 10)[0]
        print('to elements: ', x_train.shape)
        y_train = numpy.split(y_train, 10)[0]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    funcs = func_list()

    scores = []
    for i, func in enumerate(funcs):
        scores.append((func(x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test), func.__name__))
        print('\nFunction: ' + func.__name__)
        print('\tTest loss: ', scores[i][0][0])
        print('\tTest accuracy: ', scores[i][0][1])

    if (verbose):
        print('\n', scores)

    average_acc = 0
    best_acc = scores[0]
    for score in scores:
        (data, name) = score
        average_acc += data[1]
        if (data[1] > best_acc[0][1]):
            best_acc = score

    average_acc /= len(scores)
    print('\n')
    print('Average accuracy:', average_acc)
    print('Best accuracy:', best_acc[1])

    return


if __name__ == "__main__":
    main()