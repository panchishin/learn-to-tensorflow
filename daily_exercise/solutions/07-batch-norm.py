# -- imports --
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# np.set_printoptions(precision=1) reduces np precision output to 1 digit
np.set_printoptions(precision=2, suppress=True)

# -- constant data --
x = [[0., 0.], [1., 1.], [1., 0.], [0., 1.]]
y_ = [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]

# -- induction --
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = the x2 inputs
x0 = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y0 = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# Layer 1 = the 2x3 hidden sigmoid
m1 = tf.Variable(tf.random_uniform([2, 3], minval=0.1, maxval=0.9, dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([3], minval=0.1, maxval=0.9, dtype=tf.float32))
h1 = tf.sigmoid(tf.matmul(x0, m1) + b1)

# calculate the batch norm for h1
# the equation of batch normalization is ( h - average(h) ) / standard_deviation(h)
# we don't divide by the standard_deviation just in-case it is 0,
# instead divide by standard_deviation + some really small number
average1 = tf.reduce_mean(h1, 0)
std1 = tf.sqrt(tf.reduce_mean(tf.square(h1 - average1), 0))
batch_norm_h1 = (h1 - average1) / (std1 + 1e-6)

# Layer 2 = the 3x2 softmax output
m2 = tf.Variable(tf.random_uniform([3, 2], minval=0.1, maxval=0.9, dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([2], minval=0.1, maxval=0.9, dtype=tf.float32))
y_out = tf.nn.softmax(tf.matmul(batch_norm_h1, m2) + b2)

# -- loss --

# loss : sum of the squares of y0 - y_out
loss = tf.reduce_sum(tf.square(y0 - y_out))

# training step : gradient descent (1.0) to minimize loss
train = tf.train.GradientDescentOptimizer(1).minimize(loss)

# -- training --
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("\nloss")
    for step in range(500):
        sess.run(train, feed_dict={x0: x, y0: y_})
        if (step + 1) % 100 == 0:
            print(sess.run(loss, feed_dict={x0: x, y0: y_}))

    results = sess.run([m1, b1, m2, b2, y_out, loss], feed_dict={x0: x, y0: y_})
    labels = "m1,b1,m2,b2,y_out,loss".split(",")
    for label, result in zip(*(labels, results)):
        print("")
        print(label)
        print(result)

print("")

