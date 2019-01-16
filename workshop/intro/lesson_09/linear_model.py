# -- imports --
import tensorflow as tf

# -- placeholders --
x = tf.placeholder(dtype=tf.float32, shape=[None, None])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# -- variables --
# f(x) = xm + b
m = tf.Variable([[0.1], [0.2]])
b = tf.Variable(0.3)

# -- induction --
f1 = tf.matmul(x, m) + b
fx = tf.reshape(f1, [-1])

# -- loss --
# let's use RMS as our error function
rms_error = tf.sqrt(tf.reduce_mean(tf.square(fx - y)))
learn = tf.train.AdamOptimizer(0.01).minimize(rms_error)


def printEquation(sess):
    print("f(x) = x * m + b")
    print("m =", sess.run(m))
    print("b =", sess.run(b))
