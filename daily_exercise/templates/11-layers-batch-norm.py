import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.set_printoptions(precision= # TODO

# The data

x = # TODO
y = # TODO

# Define the graph

layer1 = # TODO
batch_norm = # TODO
layer2 = # TODO
model = # TODO

# Turn the graph into a model

model.compile(loss= # TODO
              optimizer= # TODO

model.fit(x, y, steps_per_epoch= # TODO

# Show the results

print("\nExpected output")
print("    ", y.reshape([-1]))

print("\nPredicted output from model")
print("    ", model.predict(x).reshape([-1]))

print()
