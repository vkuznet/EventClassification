#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import keras
from keras import backend as KKK
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='categorical_crossentropy',
                                    metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

print("dataset", type(dataset), dataset, KKK.is_tensor(dataset))
print("model.fit", type(model), type(model.fit), model.fit)
print("instance", isinstance(model, keras.engine.training.Model))

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
#model.fit(dataset, epochs=2, steps_per_epoch=30)
model.fit(dataset, epochs=2, steps_per_epoch=30, validation_data=dataset, validation_steps=1)
