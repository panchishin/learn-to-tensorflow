
# we will build on the last example and do the same thing
# but using Tensorflow

# we will define a and b as tensorflow constants

# notice that when we print them out we do not get their values,
# that is because we need to evaluate them


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
# that evaluates c

# let's evaluate a and b
print("The value of a is", sess.run(a))
print("The value of b is", sess.run(b))
