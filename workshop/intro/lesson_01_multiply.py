# -- imports --
import tensorflow as tf

# -- constants --
a = tf.constant(5.0)
b = tf.constant(7.0)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# start a session
sess = tf.Session()

# let's check out a, b, and c
print("a =", a)
print("b =", b)
print("c =", c)

# let's do the multiplication
print("The result of 5x7 is", sess.run(c))
