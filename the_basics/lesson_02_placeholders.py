
# Our code will be really limited in use if all we can use is constants
# in tensorflow

# Here we introduce how we can pass different values using placeholders
# and passing the values in using the feed_dict argument in sess.run


# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import placeholder, Session

# -- variables --
a = placeholder(dtype=tf.float32)
b = placeholder(dtype=tf.float32)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# start a session
sess = Session()

# let's do the multiplication
print("The result of 5x7 is", sess.run(c, feed_dict={a: 5, b: 7}))

# let's do the multiplication
print("The result of 2x3 is", sess.run(c, feed_dict={a: 2, b: 3}))
