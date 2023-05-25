from os import mkdir
from os.path import exists, join
from time import strftime

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import relative_path


def make_model(input_shape, num_classes, data_augmentation = False):
    inputs = keras.Input(shape=input_shape)

    if data_augmentation:
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )

        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(x)

    else:
        x = inputs

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def main():

    img_folder_path = relative_path("data", "PetImages")

    image_size = (180, 180)
    batch_size = 16

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        img_folder_path,
        validation_split = 0.2,
        subset = "both",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size,
    )

    model = make_model(input_shape=image_size + (3,), num_classes=2, data_augmentation=True)

    epochs = 25

    stats = {"epoch":[], "loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[]}

    def addstats(epoch, logs):

        for key in stats:
            if key == "epoch":
                stats["epoch"].append(epoch)
            else:
                stats[key].append(logs[key])


    saves_folder = join("saves", strftime("%d-%m-%Y_%H:%M:%S"))

    if not exists("saves"):
        mkdir("saves")

    mkdir(saves_folder)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath = join(saves_folder, "Best.keras"),
            monitor = "val_accuracy",
            save_best_only = True
            ),
        keras.callbacks.LambdaCallback(
                on_epoch_end = addstats
            )
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.save(join(saves_folder, "Last.keras"))

    pd.DataFrame(stats).to_csv(join(saves_folder, "stats.csv"), index=False)


if __name__ == '__main__':
    main()