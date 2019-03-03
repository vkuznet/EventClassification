#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()
print("dataset", type(dataset), dataset)

epochs = 2
steps = 30
vsteps = 1

try: # TPU detection
    tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
    print('### Training on TPU ###')
    # For TPU, we will need a function that returns the dataset
    ds_input_fn = lambda: dataset
    strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
    trained_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    trained_model.fit(ds_input_fn, epochs=epochs, steps_per_epoch=steps,
            validation_data=ds_input_fn, validation_steps=vsteps)
except ValueError:
    print('### Training on GPU/CPU ###')
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps,
            validation_data=dataset, validation_steps=vsteps)
