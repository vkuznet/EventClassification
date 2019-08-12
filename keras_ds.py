#!/usr/bin/env python
import traceback
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

data = np.array(np.random.random((1000, 32)), dtype=np.float32)
labels = np.array(np.random.random((1000, 10)), dtype=np.float32)
epochs = 2
steps = 30
vsteps = 1
batch_size = 10
tpu = None


# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.repeat()
print("dataset", type(dataset), dataset)

try: # TPU detection
    tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
    print('### Training on TPU ###')
except ValueError:
    traceback.print_exc()
    print('### Training on GPU/CPU ###')

model.fit(dataset, epochs=epochs, steps_per_epoch=steps,
        validation_data=dataset, validation_steps=vsteps)
