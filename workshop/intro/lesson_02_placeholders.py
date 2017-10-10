# -- imports --
import tensorflow as tf

# -- variables --
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# start a session
sess = tf.Session()

# let's do the multiplication
print "The result of 5x7 is", sess.run(c, feed_dict={a: 5, b: 7})

# let's do the multiplication
print "The result of 2x3 is", sess.run(c, feed_dict={a: 2, b: 3})
