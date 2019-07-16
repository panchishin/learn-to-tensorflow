# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.train import AdamOptimizer

# -- placeholders --
x = placeholder(dtype=tf.float32, shape=[None])
y = placeholder(dtype=tf.float32, shape=[None])

# -- variables --
# f(x) = ax + b
a = tf.Variable(0.0)
b = tf.Variable(0.0)
c = tf.Variable(0.0)

# -- induction --
fx = b * x + c

# -- loss --
# let's use RMS as our error function
rms_error = tf.sqrt(tf.reduce_mean(tf.square(fx - y)))
learn = AdamOptimizer().minimize(rms_error)


def printEquation(sess):
    print("f(x) =", sess.run(b), " * x +", sess.run(c))
