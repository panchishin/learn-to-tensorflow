# -- imports --
import tensorflow as tf
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
# but we want c to equal the target
# so we calculate a loss accordingly
loss = tf.square(c - args.target)
learn = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)

# start a session
sess = tf.Session()

# initialize the variables
sess.run(tf.global_variables_initializer())

# let's do the multiplication
print("The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c), ", but we want it to =", args.target)

for iteration in range(1, args.iterations + 1):
    sess.run(learn)
    if iteration < 10 or (iteration % 10 == 0 and iteration < 100) or (iteration % 100 == 0 and iteration < 1000) or iteration % 1000 == 0:
        print("On iteration", iteration, "the result of", sess.run(a), "x", sess.run(b), "is", sess.run(c))
