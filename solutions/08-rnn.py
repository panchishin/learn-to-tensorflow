# -- imports --
import tensorflow as tf
from helper_make_xy import makeXY

labels = 10
neurons = 20
layers = 2
steps_per_report = 500

# -- induction --

# input layer
dropout = tf.placeholder(dtype=tf.float32)
x0 = tf.placeholder(dtype=tf.int32, shape=[None, None])
y0 = tf.placeholder(dtype=tf.int32, shape=[None])

# one-hot encoding
x0_hot = tf.one_hot(x0, labels, dtype=tf.float32)
y0_hot = tf.one_hot(y0, labels, dtype=tf.float32)

# RNN layer cell definition
cells = []
for _ in range(layers):
    cell = tf.contrib.rnn.GRUCell(neurons)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    cells.append(cell)

cell = tf.contrib.rnn.MultiRNNCell(cells)

# RNN output
output, state = tf.nn.dynamic_rnn(cell, x0_hot, dtype=tf.float32)
last_output = tf.reshape(output[-1, -1], [-1, neurons])

# fully connected output layer
m = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[neurons, labels]))
b = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[labels]))
z = tf.matmul(last_output, m) + b
y_out = tf.nn.softmax(tf.reshape(z, [labels]))
y_argmax = tf.argmax(y_out, 0)


# -- loss --
# use basic cross entropy
loss = -tf.reduce_mean(tf.reduce_mean(y0_hot * tf.log(y_out)))


# -- training --
# use adam or gradient decent optimizer with 0.01
train = tf.train.AdamOptimizer().minimize(loss)


# -- Execution --
with tf.Session() as sess:
    # initialize session variables
    sess.run(tf.global_variables_initializer())

    for report in range(20):
        # initialize avgloss and num correct to 0 and 0.0 respectively
        avgloss, correct = 0, 0.0

        for step in range(steps_per_report):
            # retrieve x and y example data using makeXY()
            inx, outy = makeXY()

            # execute 'train' with dropout of 0.5 to train a resilient NN
            sess.run(train, feed_dict={x0: [inx], y0: [outy], dropout: 0.5})

            # execute 'loss' and 'y_argmax' with dropout of 1.0 to gather stats
            thisloss, result = sess.run([loss, y_argmax], feed_dict={x0: [inx], y0: [outy], dropout: 1.0})
            # add thisloss to avgloss
            avgloss += thisloss
            # increment correct if result is the same as outy
            correct += 1.0 if result == outy else 0.0

        print("Report =", (report + 1), " , correct =", (correct / steps_per_report), " , avgloss =", (avgloss / steps_per_report))
        if correct == steps_per_report:
            print("Finished early.")
            break
