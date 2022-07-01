import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging

# Functions
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# Code

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Import dataset
tfds.disable_progress_bar()
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Labels, for plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))

# Normalize values of pixels (applies normalize function to every element, returns the final list)
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Keep dataset in cache to make things faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28,28))


# Build the network
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=15, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCH_SIZE))

print("Accuracy n test dataset: ", test_accuracy)






