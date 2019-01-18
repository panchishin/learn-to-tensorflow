import tensorflow as tf
import numpy as np
import os
import sys

# np.set_printoptions(precision=1) reduces np precision output to 1 digit
np.set_printoptions(precision=2, suppress=True)

# placeholders
data   = tf.placeholder( shape=[None,2], dtype=tf.float32)
labels = tf.placeholder( shape=[None,2], dtype=tf.float32)

# data -> hidden layer with 3 units and tanh activation
# hidden -> prediction layer with softmax activation
hidden = tf.layers.dense( inputs=data, units=3, activation=tf.nn.tanh, name='hidden')
predict = tf.layers.dense( inputs=hidden, units=2, activation=tf.nn.softmax, name="predict" )

# loss using the mean of the squares of the error
loss = tf.reduce_mean( tf.square( labels - predict ) )

# training using gradient descent with a learning rate of 1
train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

feed_dict = {
    data   : [[0.,0.],[1.,1.],[1.,0.],[0.,1.]],
    labels : [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]
    }

# -- training --
# run 500 times using all the data and labels
# print out the loss and any other interesting info
file_name = "./temp7.save"
with tf.Session() as sess:
    saver = tf.train.Saver()
    try :
        saver.restore(sess, file_name)
        print("Restored from file")
    except :
        print("No save file found")
        sess.run(tf.global_variables_initializer())
        print("\ntraining loss")
        for step in range(500):
            sess.run(train, feed_dict=feed_dict)
            if (step + 1) % 100 == 0:
                print(sess.run(loss, feed_dict=feed_dict))
        saver.save(sess, file_name)

    results = sess.run([loss,labels,predict], feed_dict=feed_dict)
    labels = "loss,labels,predict".split(",")
    for label, result in zip(*(labels, results)):
        print("")
        print(label)
        print(result)