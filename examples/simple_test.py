import tensorflow as tf
import harness
import models
import layer


class simple:

    def convolve(self, image, training,  keep_prob):
        result = image
        result = layer.batch_normalization(result, training)
        result = layer.conv(result, 1, 16, width=5, stride=2, padding="VALID")
        result = tf.nn.tanh(result)
        result = layer.conv(result, 16, 16, width=3, stride=2, padding="VALID")
        result = tf.nn.tanh(result)
        result = layer.conv(result, 16, 32, width=3, padding="VALID")
        result = tf.nn.tanh(result)
        result = layer.conv(result, 32, 32, width=3, padding="VALID")
        result = tf.nn.tanh(result)
        result = tf.nn.dropout(result, keep_prob)
        result = layer.conv_relu(result, 32, 10, width=1, padding="VALID")
        return result


harness.train_model_and_report(simple(), epochs=100, learning_rate_value=1e-3)
