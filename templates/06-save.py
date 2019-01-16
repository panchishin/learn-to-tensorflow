# -- imports --
import tensorflow as tf
import os
import numpy as np

# np.set_printoptions(precision= # TODO
np.set_printoptions(precision= # TODO

# file name
file_name = # TODO

# -- constant data --
x = # TODO
y_ = # TODO

# -- induction --
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = # TODO
x0 = # TODO
y0 = # TODO

# Layer 1 = # TODO
m1 = # TODO
b1 = # TODO
h1 = # TODO

# Layer 2 = # TODO
m2 = # TODO
b2 = # TODO
y_out = # TODO


# -- loss --

# loss : sum of the squares of y0 - y_out
loss = # TODO

# training step : gradient decent (1.0) to minimize loss
train = # TODO


# -- training --
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:

    if not os.path.exists(file_name + ".index"):
        # TODO session execution command here
        print("No training file found.  Training...")
        print("\nloss")
        for step in range(500):
            # TODO session execution command here
            if (step + 1) % 100 = # TODO
                print(# TODO session execution command here
        print("Training complete.  Saving...")
        saver = # TODO
        saver.save(sess, file_name)
        print("Model saved to file", file_name)
        print("Run program again to use model.")

    else:
        print("Training file", file_name, "found.")
        saver = # TODO
        saver.restore(sess, file_name)
        results = # TODO
        labels = # TODO
        for label, result in zip(*(labels, results)):
            print("")
            print(label)
            print(result)

print("")
