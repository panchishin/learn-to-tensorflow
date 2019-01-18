# -- imports --
import tensorflow as tf
import numpy as np

# np.set_printoptions(precision=1) reduces np precision output to 1 digit
np.set_printoptions(precision= # TODO

# -- constant data --
x = # TODO
y_ = # TODO

# -- induction --
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = the x2 inputs
x0 = # TODO
y0 = # TODO

# Layer 1 = the 2x3 hidden sigmoid
m1 = # TODO
b1 = # TODO
h1 = # TODO

# Layer 2 = the 3x1 sigmoid output
m2 = # TODO
b2 = # TODO
y_out = # TODO


# -- loss --

# loss : sum of the squares of y0 - y_out
loss = # TODO

# training step : gradient descent (1.0) to minimize loss
train = # TODO


# -- training --
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
    # TODO session execution command here
    print("\nloss")
    for step in range(500):
        # TODO session execution command here
        if (step + 1) % 100 = # TODO
            print(# TODO session execution command here

    results = # TODO
    labels = # TODO
    for label, result in zip(*(labels, results)):
        print("")
        print(label)
        print(result)

print("")
