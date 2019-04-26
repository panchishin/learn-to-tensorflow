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
# In machine learning, we often call the error function the 'loss'.
# We want c to equal the target so we calculate a loss accordingly.
# In this case we will use the square of the difference as the error.
loss = tf.square(c - args.target)

# We want tensorflow to learn from the loss (the error) and
# this is how do that
optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
learn = optimizer.minimize(loss)

# we want to see what the optimizer is doing in this example,
# so we will get the computed gradients (error correction) from the optimizer and display them.
# usually we don't do this, but it will help us see what is going on
gradient = optimizer.compute_gradients(loss, var_list=[a])


# start a session
sess = tf.Session()

# initialize the variables
sess.run(tf.global_variables_initializer())

# let's do the multiplication
print("The result of ", sess.run(a), "x", sess.run(b), "is", sess.run(c), ", but we want it to =", args.target)

print()
print("We will use tensorflow to 'learn' the variable a.")
print()

print('-'*60)
print('Iteration  | gradient * learning rate | Result a * b = c       ')
print('           |     = correction to a    |')
print('-'*60)
print(f'{0:9}  |                          | {sess.run(a):.2f} * {sess.run(b):.2f} = {sess.run(c):.2f}')
for iteration in range(1, args.iterations + 1):

    # learn a
    sess.run(learn)

    # print out what is going on
    if iteration < 10 or (iteration % 10 == 0 and iteration < 100) or (iteration % 100 == 0 and iteration < 1000) or iteration % 1000 == 0:
        print(f'{iteration:9}  | {sess.run(gradient)[0][0]:5.2f} * {args.learning_rate:.4f} = {sess.run(gradient)[0][0]*args.learning_rate:.4f}  | {sess.run(a):.2f} * {sess.run(b):.2f} = {sess.run(c):.2f}')

print()
print("As you can see, the value of 'a' changes by the gradient * learning_rate at each iteration")
print()
print("The code : ")
print()
print(">   optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)")
print(">   learn = optimizer.minimize(loss)")
print()
print("Tells tensorflow to use the error function (loss) to derive gradient and")
print("create a function (learn) that will minimize the error.")
print("We could use the value of gradient to do this ourselves, but we")
print("well see that for more complicated graphs the above 2 lines does is more concise,")
print("and infact we can reduce it to one line")
print()
print(">   learn = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)")
print()