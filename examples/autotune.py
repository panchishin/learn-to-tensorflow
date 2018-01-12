import tensorflow as tf
import layer
import numpy as np


class AutoTune:

    def __init__(self, variables=5, expanse=4):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with self.graph.container("tuner"):
                self.x_in = tf.placeholder(tf.float32, [None, variables])
                self.y_in = tf.placeholder(tf.float32, [None])

                h = layer.fully_connected(self.x_in, variables, variables * expanse, name="tuner")
                h = tf.nn.tanh(h)
                h = layer.fully_connected(h, variables * expanse, variables * expanse, name="tuner")
                h = tf.nn.tanh(h)
                h = layer.fully_connected(h, variables * expanse, 1, name="tuner")

                self.y_out = tf.reshape(h, [-1])

                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_out, labels=self.y_in))
                self.train = tf.train.GradientDescentOptimizer(.1).minimize(self.loss)

                self.choice = tf.nn.softmax(self.y_out)

        self.sess = tf.Session(graph=self.graph)
        self.reinit()

    def reinit(self):
        with self.graph.as_default():
            for item in tf.trainable_variables():
                self.sess.run(item.initializer)

    def optimize(self, training_data, y_data, iterations=100, batch_size=200):
        for count in range(iterations):
            batches = (y_data.shape[0] - 1) / batch_size
            for batch in range(batches + 1):
                end = y_data.shape[0] - batch_size * batch
                start = max(0, end - batch_size)
                self.sess.run(self.train, feed_dict={self.x_in: training_data[start:end, :], self.y_in: y_data[start:end]})

    def chance(self, decision_data):
        return self.sess.run(self.choice, feed_dict={self.x_in: decision_data})

    def choose(self, chance):
        return np.random.choice(chance.shape[0], 1, p=chance)[0]


if __name__ == '__main__':
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

    decision_data = prepare_data([
        [.032, .022, .1, 200],
        [.032, .022, .2, 100],
        [.032, .022, .015, 200],
        [.032, .022, .012, 100]
    ])

    tuner = AutoTune()
    print
    print "Just initialized"
    chance = tuner.chance(decision_data)
    print "Chances", chance, "chose option", tuner.choose(chance)
    print
    print "Optimize with first 7 data points"
    tuner.optimize(training_data[:, :7], y_data)
    chance = tuner.chance(decision_data)
    print "Chances", chance, "chose option", tuner.choose(chance)
    print
    print "Optimize with all data"
    tuner.optimize(training_data, y_data, iterations=100)
    chance = tuner.chance(decision_data)
    print "Chances", chance, "chose option", tuner.choose(chance)
    print
    print "Reinitialize"
    tuner.reinit()
    chance = tuner.chance(decision_data)
    print "Chances", chance, "chose option", tuner.choose(chance)
    print
