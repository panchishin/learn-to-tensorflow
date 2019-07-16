# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.train import AdamOptimizer

# -- placeholders --
x = placeholder(dtype=tf.float32, shape=[None, None])
y = placeholder(dtype=tf.float32, shape=[None])


# -- variables and induction --
# -- layer 1 --
# L1(x) = tanh( x * m1 )
m1 = tf.Variable(tf.random_uniform(shape=[2, 3], dtype=tf.float32))
L1 = tf.matmul(x, m1)
L1 = tf.tanh(L1)
# -- layer 2 --
# L2(x) = L1 * m2
m2 = tf.Variable(tf.random_uniform(shape=[3, 1], dtype=tf.float32))
L2 = tf.matmul(L1, m2)

# reshape output with is [?,1] to be [?]
fx = tf.reshape(L2, [-1])

# -- loss --
rms_error = tf.sqrt(tf.reduce_sum(tf.square(y - fx)))

learn = AdamOptimizer(.01).minimize(rms_error)


def printEquation(sess):
    print("f(x) = tanh( x * m1 ) * m2")
    print("m1 =")
    print(sess.run(m1))
    print("m2 =")
    print(sess.run(m2))
