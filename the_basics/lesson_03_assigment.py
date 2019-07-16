# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import Session, global_variables_initializer, assign

# -- variables -- (we'll refer to constants as variables from now on)
a = tf.Variable(5.0)
b = tf.constant(7.0)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# start a session
sess = Session()

# we have to initialize variables
sess.run(global_variables_initializer())

# let's do the multiplication
print("The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c))

sess.run(assign(a, 10))
print("The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c))
