import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow import io
from tensorflow import image
import cv2 as cv

from pathlib import Path

batch_size = 32
img_height = 180
img_width = 180


def create_model():
    data_dir = pathlib.Path(".\\nl_birds")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(train_ds)

    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.6),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(7)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save('my_model.h5')


def test_model(img_name):
    model = keras.models.load_model('.\\my_model.h5')

    img = tf.keras.utils.load_img(img_name, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    bird_class_names = ["Cardinal", "Crow", "Grouse", "Morning Warbler", "Puffin", "Seagull", "Tree Sparrow"]

    print(predictions)
    print(bird_class_names[np.argmax(score)])
    display_img_and_prediction(bird_class_names[np.argmax(score)], img_name)


def display_img_and_prediction(prediction, img):
    img = cv.imread(img, cv.IMREAD_ANYCOLOR)

    # font
    font = cv.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    resized = cv.resize(img, (500,300), interpolation=cv.INTER_LINEAR)
    h, w, c = resized.shape
    # org
    org = (w - 475, h - 30)
    resized = cv.putText(resized, str(prediction), org, font, fontScale, color, thickness, cv.LINE_AA)

    while True:
        cv.imshow("Bird", resized)
        cv.waitKey(0)
        sys.exit()  # to exit from all the processes

    cv.destroyAllWindows()  # destroy all windows

if (len(sys.argv) >= 2 and sys.argv[1] == "create"):
    create_model()
elif (len(sys.argv) >= 2 and sys.argv[1] == "test"):
    test_model(sys.argv[2])
else:
    print(
        "\nUsages:\n\nCommand: bird.py create => Creates the CNN for the bird image classifier using the nl_birds dataset.")
    print("Saves to the working directory as \"my_model.h5\"\n")
    print(
        "Command: bird.py test <image_name> => Prints prediction using the model and the image from <image_name> as input.\n")
