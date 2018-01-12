import tensorflow as tf
import layer
import numpy as np


def prepare_data(data_in):
    x = np.array(data_in, dtype=np.float32)
    y = x[:, 0:1] / x[:, 1:2]
    z = np.concatenate([x, y], 1)
    return np.log(z, dtype=np.float32)

training_data = prepare_data([
    [.032, .022, .15, 200],
    [.031, .027, .2, 100],
    [.042, .040, .15, 200],
    [.022, .022, .12, 100],
    [.0321, .022, .01, 200],
    [.0032, .022, .16, 100],
    [.32, .32, .1, 200],
    [.32, .2, .12, 100]
])

y_data = np.array([1.0 if x[2] > x[1] else 0.0 for x in training_data], dtype=np.float32)
print y_data

decision_data = prepare_data([
    [.032, .022, .1, 200],
    [.032, .022, .2, 100],
    [.032, .022, .015, 200],
    [.032, .22, .12, 100]
])

w = layer.weight_variable([10, 10])

graph = tf.Graph()

with graph.as_default():
    with graph.container("tuner"):
        x_in = tf.placeholder(tf.float32, [None, 5])
        y_in = tf.placeholder(tf.float32, [None])

        h = layer.fully_connected(x_in, 5, 5, name="tuner")
        h = tf.nn.tanh(h)
        h = layer.fully_connected(h, 5, 5, name="tuner")
        h = tf.nn.tanh(h)
        h = layer.fully_connected(h, 5, 1, name="tuner")

        y_out = tf.reshape(h, [-1])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_in))
        train = tf.train.GradientDescentOptimizer(.1).minimize(loss)

        choice = tf.nn.softmax(y_out)

        init_op = tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
    sess.run(init_op)

    print "y_out = ", sess.run(y_out, feed_dict={x_in: training_data, y_in: y_data})

    choice_result = sess.run(choice, feed_dict={x_in: training_data, y_in: y_data})
    a_choice = np.random.choice(choice_result.shape[0], 1, p=choice_result)[0]
    print "choices =", choice_result
    print "a choice =", a_choice

    for count in range(1, 1001):
        if count == 1 or count == 10 or count == 100 or count == 1000:
            print "loss = ", sess.run(loss, feed_dict={x_in: training_data, y_in: y_data})
        sess.run(train, feed_dict={x_in: training_data, y_in: y_data})

    choice_result = sess.run(choice, feed_dict={x_in: decision_data})
    a_choice = np.random.choice(choice_result.shape[0], 1, p=choice_result)[0]
    print "choices =", choice_result
    print "a choice =", a_choice

    for item in tf.trainable_variables():
        # sess.run(item.initializer)
        print item
        # print sess.run(item)
