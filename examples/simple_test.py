import tensorflow as tf
import harness
import models
import layer


class simple:

    def convolve(self, image, training,  keep_prob):
        result = image
        result = layer.conv(result, 1, 32, width=28, padding="VALID")
        result = tf.nn.tanh(result)
        result = layer.conv(result, 32, 64, width=1, padding="VALID")
        result = tf.nn.tanh(result)
        result = tf.nn.dropout(result, keep_prob)
        result = layer.conv_relu(result, 64, 10, width=1, padding="VALID")
        return result


harness.train_model_and_report(simple(), epochs=100, learning_rate_value=1e-3)
