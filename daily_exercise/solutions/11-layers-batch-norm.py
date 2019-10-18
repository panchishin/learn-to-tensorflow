import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.set_printoptions(precision=3, suppress=True)

# The data

x = np.array([[0., 0.], [1., 1.], [1., 0.], [0., 1.]])
y = np.array([[0.], [0.], [1.], [1.]])

# Define the graph

layer1 = tf.keras.layers.Dense(3, activation='sigmoid')
batch_norm = tf.keras.layers.BatchNormalization()
layer2 = tf.keras.layers.Dense(1, activation='sigmoid')
model = tf.keras.Sequential([layer1, batch_norm, layer2])

# Turn the graph into a model

model.compile(loss=tf.keras.metrics.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(1))

model.fit(x, y, steps_per_epoch=500)

# Show the results

print("\nExpected output")
print("    ", y.reshape([-1]))

print("\nPredicted output from model")
print("    ", model.predict(x).reshape([-1]))

print()
