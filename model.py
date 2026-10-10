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
class_names = train_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
validate_dataset = validate_dataset.prefetch(AUTOTUNE)

model = tf.keras.Sequential([
    layers.Input(shape=(128,128,3), batch_size=32),
    layers.Rescaling(1.0/255),
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    
    # Layer 1     
    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    ),  
    layers.MaxPool2D(
        pool_size=(2, 2),

    ),

    # Layer 2 
    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    ),  
    layers.MaxPool2D(
        pool_size=(2, 2),
    ),

    # Layer 3
    layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        ),  
        layers.MaxPool2D(
            pool_size=(2, 2),
    ),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5)
])

print(model.summary())



