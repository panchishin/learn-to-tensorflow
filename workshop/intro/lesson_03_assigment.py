# -- imports --
import tensorflow as tf

# -- variables -- (we'll refer to constants as variables from now on)
a = tf.Variable(5.0)
b = tf.constant(7.0)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# start a session
sess = tf.Session()

# we have to initialize variables
sess.run(tf.global_variables_initializer())

# let's do the multiplication
print "The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c)

sess.run(tf.assign(a, 10))
print "The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c)
