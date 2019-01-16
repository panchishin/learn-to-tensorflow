# -- imports --
import tensorflow as tf


# -- variables --
# f(x) = ax + b
a = tf.Variable(0.0)
x = tf.placeholder(dtype=tf.float32, shape=[None])
b = tf.Variable(0.0)
c = tf.Variable(0.0)


# -- induction --
# f(x) = ax^2 +bx + c
fx = a * tf.square(x) + b * x + c

# -- loss --
# but we want f(x) to equal y
y = tf.placeholder(dtype=tf.float32, shape=[None])
# so we calculate a loss accordingly
# loss = tf.reduce_mean(tf.square(fx - y))
loss = tf.sqrt(tf.reduce_mean(tf.square(fx - y)))
learn = tf.train.GradientDescentOptimizer(.0001).minimize(loss)

# start a session
sess = tf.Session()
# initialize the variables
sess.run(tf.global_variables_initializer())

data_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
data_y = [1.0, 2.0, 4.0, 4.0, 6.0, 5.0, 7.0, 7.0, 9.0, 10.0, 9.50, 8.00, 7.00, 6.00, 4.00, 5.00, 3.00, 2.00, 1.00]


print("Calling get better")
for iteration in range(1, 10001):
    sess.run(learn, feed_dict={x: data_x, y: data_y})
    if iteration == 1 or iteration == 10 or iteration == 100 or iteration % 1000 == 0:
        print("iteration", iteration,)
        print(", the average loss =", sess.run(loss, feed_dict={x: data_x, y: data_y}))

print("The equation is f(x) =", sess.run(a), "* x^2 +", sess.run(b), "* x +", sess.run(c))
