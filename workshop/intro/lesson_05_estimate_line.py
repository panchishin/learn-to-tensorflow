# -- imports --
import tensorflow as tf


# -- variables --
# f(x) = a * x + b
a = tf.Variable(0.0)
x = tf.placeholder(dtype=tf.float32)
b = tf.Variable(0.0)

# -- induction --
# f(x) = a * x + b
fx = tf.add(tf.multiply(a, x), b)

# -- loss --
# but we want f(x) to equal y
y = tf.placeholder(dtype=tf.float32)
# so we calculate a loss accordingly
loss = tf.square(fx - y)
learn = tf.train.GradientDescentOptimizer(.001).minimize(loss)

# start a session
sess = tf.Session()
# initialize the variables
sess.run(tf.global_variables_initializer())

data_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
data_y = [0.1, 0.2, 0.4, 0.4, 0.6, 0.5, 0.7, 0.7, 0.9]

# let's calculate the total loss


def printTotalLoss():
    total_loss = 0
    for index in range(len(data_x)):
        total_loss += sess.run(loss, feed_dict={x: data_x[index], y: data_y[index]})
    print("The total loss is", total_loss)


printTotalLoss()


def getBetter():
    for index in range(len(data_x)):
        sess.run(learn, feed_dict={x: data_x[index], y: data_y[index]})


print("Calling get better")
for iteration in range(1, 1001):
    getBetter()
    if iteration == 1 or iteration == 10 or iteration == 100 or iteration == 1000:
        print("iteration", iteration,)
        printTotalLoss()

print("The equation is f(x) =", sess.run(a), "* x +", sess.run(b))
