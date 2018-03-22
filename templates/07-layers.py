import tensorflow as tf
import numpy as np
import os
import sys

# np.set_printoptions(precision= # TODO
np.set_printoptions(precision= # TODO

# placeholders
data   = # TODO
labels = # TODO

# data -> hidden layer with 3 units and tanh activation
# hidden -> prediction layer with softmax activation
hidden = # TODO
predict = # TODO

# loss using the mean of the squares of the error
loss = # TODO

# training using gradient descent with a learning rate of 1
train = # TODO

feed_dict = # TODO
    data   : [[0.,0.],[1.,1.],[1.,0.],[0.,1.]],
    labels : [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]
    }

# -- training --
# run 500 times using all the data and labels
# print out the loss and any other interesting info
file_name = # TODO
with tf.Session() as sess:
    saver = # TODO
    try :
        saver.restore(sess, file_name)
        print "Restored from file"
    except :
        print "No save file found"
        # TODO session execution command here
        print "\ntraining loss"
        for step in range(500):
            # TODO session execution command here
            if (step + 1) % 100 = # TODO
                print # TODO session execution command here
        saver.save(sess, file_name)

    results = # TODO
    labels = # TODO
    for label, result in zip(*(labels, results)):
        print ""
        print label
        print result
