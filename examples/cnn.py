import argparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def evaluate(args):
    shuffle = np.random.permutation(50000)
    i_train = shuffle[:40000]
    i_validation = shuffle[40000:]
    (train_images, train_labels), _ = datasets.cifar10.load_data()
    validation_images = train_images[i_validation]
    validation_labels = train_labels[i_validation]
    train_images = train_images[i_train]
    train_labels = train_labels[i_train]

    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    if args.num_blocks >= 1:
        model.add(
            layers.Conv2D(args.num_filters_1, (3, 3), activation="relu", padding="same")
        )
        model.add(
            layers.Conv2D(args.num_filters_1, (3, 3), activation="relu", padding="same")
        )
        if args.batch_norm_1:
            model.add(layers.BatchNormalization())
        if args.pooling == "Max":
            model.add(layers.MaxPooling2D((2, 2)))
        elif args.pooling == "Average":
            model.add(layers.AveragePooling2D((2, 2)))
        model.add(layers.Dropout(args.dropout_rate))
    if args.num_blocks >= 2:
        model.add(
            layers.Conv2D(args.num_filters_2, (3, 3), activation="relu", padding="same")
        )
        model.add(
            layers.Conv2D(args.num_filters_2, (3, 3), activation="relu", padding="same")
        )
        if args.batch_norm_2:
            model.add(layers.BatchNormalization())
        if args.pooling == "Max":
            model.add(layers.MaxPooling2D((2, 2)))
        elif args.pooling == "Average":
            model.add(layers.AveragePooling2D((2, 2)))
        model.add(layers.Dropout(args.dropout_rate))
    if args.num_blocks >= 3:
        model.add(
            layers.Conv2D(args.num_filters_3, (3, 3), activation="relu", padding="same")
        )
        model.add(
            layers.Conv2D(args.num_filters_3, (3, 3), activation="relu", padding="same")
        )
        if args.batch_norm_3:
            model.add(layers.BatchNormalization())
        if args.pooling == "Max":
            model.add(layers.MaxPooling2D((2, 2)))
        elif args.pooling == "Average":
            model.add(layers.AveragePooling2D((2, 2)))
        model.add(layers.Dropout(args.dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(args.num_units, activation="relu"))
    model.add(layers.Dropout(args.dropout_rate))
    model.add(layers.Dense(10, activation="softmax"))
    model.summary()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        nesterov=False,
        name="SGD",
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_images, train_labels, batch_size=32, epochs=50, shuffle=True, verbose=0
    )

    # f1: prediction error rate
    validation_loss, validation_acc = model.evaluate(
        validation_images, validation_labels, verbose=0
    )
    validation_error = 1 - validation_acc

    # f2: average prediction speed
    elapsed_time = 0
    n_times = 10
    for i in range(n_times):
        slice_start = 1000 * i
        slice_end = 1000 * (i + 1)
        start = time.time()
        model.predict(
            validation_images[slice_start:slice_end], batch_size=1000, verbose=0
        )
        end = time.time()
        elapsed_time += end - start
    elapsed_time /= n_times

    return validation_error, elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_blocks", default=1, type=int)  # uniform integer [1, 3]
    parser.add_argument(
        "--num_filters_1", default=16, type=int
    )  # conditional uniform integer [16, 256]
    parser.add_argument(
        "--num_filters_2", default=16, type=int
    )  # conditional uniform integer [16, 256]
    parser.add_argument(
        "--num_filters_3", default=16, type=int
    )  # conditional uniform integer [16, 256]
    parser.add_argument(
        "--batch_norm_1", default=1, type=int
    )  # conditional categorical [0, 1] (0: False, 1: True)
    parser.add_argument(
        "--batch_norm_2", default=1, type=int
    )  # conditional categorical [0, 1] (0: False, 1: True)
    parser.add_argument(
        "--batch_norm_3", default=1, type=int
    )  # conditional categorical [0, 1] (0: False, 1: True)
    parser.add_argument(
        "--pooling", default="Max", type=str
    )  # categorical ['Max', 'Average']
    parser.add_argument(
        "--dropout_rate", default=0.5, type=float
    )  # uniform real [0, 0.9]  # ignore: f2
    parser.add_argument(
        "--num_units", default=16, type=int
    )  # log uniform integer [16, 4096]
    parser.add_argument(
        "--learning_rate", default=0.001, type=float
    )  # log uniform real [0.00001, 0.1]  # ignore: f2
    parser.add_argument(
        "--momentum", default=0.9, type=float
    )  # uniform real [0.8, 1.0]  # ignore: f2
    args = parser.parse_args()

    print(args)

    validation_error, elapsed_time = evaluate(args)
    print(validation_error, elapsed_time, flush=True)
