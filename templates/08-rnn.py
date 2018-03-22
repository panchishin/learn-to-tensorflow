# -- imports --
import tensorflow as tf
from helper_make_xy import makeXY

labels = # TODO
neurons = # TODO
layers = # TODO
steps_per_report = # TODO

# -- induction --

# input layer
dropout = # TODO
x0 = # TODO
y0 = # TODO

# one-hot encoding
x0_hot = # TODO
y0_hot = # TODO

# RNN layer cell definition
cells = # TODO
for _ in range(layers):
    cell = # TODO
    cell = # TODO
    cells.append(cell)

cell = # TODO

# RNN output
output, state = # TODO
last_output = # TODO

# fully connected output layer
m = # TODO
b = # TODO
z = # TODO
y_out = # TODO
y_argmax = # TODO


# -- loss --
# use basic cross entropy
loss = # TODO


# -- training --
# use adam or gradient decent optimizer with 0.01
train = # TODO


# -- Execution --
with tf.Session() as sess:
    # initialize session variables
    # TODO session execution command here

    for report in range(20):
        # initialize avgloss and num correct to 0 and 0.0 respectively
        avgloss, correct = # TODO

        for step in range(steps_per_report):
            # retrieve x and y example data using makeXY()
            inx, outy = # TODO

            # execute 'train' with dropout of 0.5 to train a resilient NN
            # TODO session execution command here

            # execute 'loss' and 'y_argmax' with dropout of 1.0 to gather stats
            thisloss, result = # TODO
            # add thisloss to avgloss
            avgloss += # TODO
            # increment correct if result is the same as outy
            correct += # TODO

        print "Report = # TODO
        if correct = # TODO
            print "Finished early."
            break
