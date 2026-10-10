import pandas as pd
import tensorflow as tf 
import keras.layers as layers 
import keras.utils as utils

train_dataset = utils.image_dataset_from_directory(
    "dataset/train",
    label_mode='int',
    batch_size=32,
    image_size=(64,64),
)

validate_dataset = utils.image_dataset_from_directory(
    "dataset/validation",
    label_mode='int',
    batch_size=32,
    image_size=(64,64),
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
validate_dataset = validate_dataset.prefetch(AUTOTUNE)
class_names = train_dataset.class_names

