# -- imports --
import tensorflow as tf

# -- placeholders --
x = tf.placeholder(dtype=tf.float32, shape=[None])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# -- variables --
# f(x) = ax + b
a = tf.Variable(0.0)
b = tf.Variable(0.0)
c = tf.Variable(0.0)

# -- induction --
# f(x) = a|x| + bx + c
fx = -1 * tf.abs(x + a) + b * x + c

# -- loss --
# let's use RMS as our error function
rms_error = tf.sqrt(tf.reduce_mean(tf.square(fx - y)))
learn = tf.train.AdamOptimizer(0.01).minimize(rms_error)


def printEquation(sess):
    print("f(x) = -1 * abs( x +", sess.run(a), ") +", sess.run(b), " * x +", sess.run(c))
