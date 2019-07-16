# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import Session, global_variables_initializer
from tensorflow.compat.v1.train import GradientDescentOptimizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', nargs='?', type=float, default=0.001, help="The learning rate.  Default to 0.001.")
parser.add_argument('--iterations', nargs='?', type=int, default=100, help="The number of iterations.  Defaults to 100.")
parser.add_argument('--target', nargs='?', type=float, default=30., help="The learning rate.  Default to 30.")
args = parser.parse_args()


# -- variables --
a = tf.Variable(5.0)
b = tf.constant(7.0)

# -- induction --
# Multiply a by b
c = tf.multiply(a, b)

# -- loss --
# In machine learning, we often call the error function the 'loss'.
# We want c to equal the target so we calculate a loss accordingly.
# In this case we will use the square of the difference as the error.
loss = tf.square(c - args.target)

# We want tensorflow to learn from the loss (the error) and
# this is how do that
optimizer = GradientDescentOptimizer(args.learning_rate)
learn = optimizer.minimize(loss)

# start a session
sess = Session()

# initialize the variables
sess.run(global_variables_initializer())

# let's do the multiplication
print("The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c), ", but we want it to =", args.target)

print()
print("We will use tensorflow to 'learn' the variable a.")
print()

print('-'*40)
print('Iteration  | Result a*b=c')
print('-'*40)
for iteration in range(1, args.iterations + 1):

    # learn a
    sess.run(learn)

    # print out what is going on
    if iteration < 10 or (iteration % 10 == 0 and iteration < 100) or (iteration % 100 == 0 and iteration < 1000) or iteration % 1000 == 0:
        print(f'{iteration:9}  | {sess.run(a):.2f} * {sess.run(b):.2f} = {sess.run(c):.2f}')

print()
