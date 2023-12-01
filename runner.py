# fmt: off
import os

# set this to 3 so tensorflow doesnt spam terminal if not GPU is present
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys

import keras
import numpy as np
import numpy.typing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, GaussianNoise, MaxPooling2D)
from keras.models import Sequential
from skimage.util import random_noise

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
    if arg == "verbose":
        verbose = True
    if arg == "debug":
        debug = True

# Models


def gaussian_noise():
    # Gaussian Noise has a regularization effect to a certain limit
    model = Sequential()
    model.add(
        Conv2D(
            16, kernel_size=(3, 3), padding="same", input_shape=(img_rows, img_cols, 1)
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GaussianNoise(0.1))
    model.add(Dense(128, activation="relu"))
    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))

    return model


def base_model():
    model = Sequential()
    model.add(
        Conv2D(
            16, kernel_size=(3, 3), padding="same", input_shape=(img_rows, img_cols, 1)
        )
    )
    model.add(Activation(keras.activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# Func list to run, add new funcs as they are made


def func_list():
    funcs = [base_model, gaussian_noise]
    # funcs = [gaussian_noise]
    return funcs


# Main function


def save_dataset(dataset, name: str) -> None:
    """
    Save a dataset in parquet format.

    Args:
        dataset: The dataset
        name (str): Name of the file
    """
    dataset = dataset.reshape((dataset.shape[0], 784))
    pa_table = pa.table(pd.DataFrame(dataset))
    pq.write_table(pa_table, f"{name}.parquet")
    return


def load_dataset(path: str) -> numpy.typing.ArrayLike:
    """
    Load a previously saved dataset from the given path.

    Args:
        path (str): The path

    Returns:
        numpy.typing.ArrayLike: The loaded dataset
    """
    pa_table = pq.read_table(path)
    return pa_table.to_pandas().to_numpy().reshape((pa_table.shape[0], 28, 28))


def add_noise(dataset, noise_type: str, amount: float = 0.05) -> numpy.typing.ArrayLike:
    """
    Adds noise to the provided dataset.
    Source: https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

    Args:
        dataset: MNIST images
        noise_type (str): Noise type, valid noise types: "gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"
        amount (float, optional): The amount of noise. Defaults to 0.05.

    Returns:
        numpy.typing.ArrayLike: The dataset with added noise
    """
    return np.array(
        [np.array(random_noise(img, mode=noise_type, amount=amount)) for img in dataset]
    )


def generate_noisy_datasets(dataset) -> None:
    """
    Generate parquet files for all noise types and amounts.

    Args:
        dataset: The clean dataset.
    """
    # noise_types = ["gaussian", "speckle"]
    s_p_noise_types = ["salt", "pepper", "s&p"]
    noise_amounts = [0.05, 0.1, 0.25]
    for noise_amount in noise_amounts:
        for noise_type in s_p_noise_types:
            print(f"Generating dataset {noise_type=}, {noise_amount=}")
            noisy_dataset = add_noise(
                dataset, noise_type=noise_type, amount=noise_amount
            )
            save_dataset(noisy_dataset, f"mnist_{noise_type}_{noise_amount}")


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if debug:
        print("elements reduced from: ", x_train.shape)
        x_train = numpy.split(x_train, 10)[0]
        print("to elements: ", x_train.shape)
        y_train = numpy.split(y_train, 10)[0]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    funcs = func_list()

    scores = []
    for i, func in enumerate(funcs):
        model = func()

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        model.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0
        )

        if verbose:
            model.summary()

        score = model.evaluate(x_test, y_test, verbose=0)

        scores.append((score, func.__name__))
        print("\nFunction: " + func.__name__)
        print("\tTest loss: ", scores[i][0][0])
        print("\tTest accuracy: ", scores[i][0][1])

    if verbose:
        print("\n", scores)

    average_acc = 0
    best_acc = scores[0]
    for score in scores:
        (data, name) = score
        average_acc += data[1]
        if data[1] > best_acc[0][1]:
            best_acc = score

    average_acc /= len(scores)
    print("\n")
    print("Average accuracy:", average_acc)
    print("Best accuracy:", best_acc[1])

    print("{0:^20}|{1:^10}".format("name", "deviation (from avg)"))
    for score in scores:
        (data, name) = score
        print("{0:^20}|{1:^10.3f}".format(name, data[1] / average_acc))

    return


if __name__ == "__main__":
    main()
