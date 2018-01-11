import tensorflow as tf

execfile("layer.py")


class simple0:

    def convolve(self, image, training,  keep_prob):
        return conv(image, 1, 10, width=28, padding="VALID")


class simple1:

    def convolve(self, image, training,  keep_prob):
        result = conv(image, 1, 18, width=5, stride=2, padding="VALID")
        return conv(result, 18, 10, width=12, padding="VALID")


class simple2:

    def convolve(self, image, training,  keep_prob):
        result = conv_relu(image, 1, 18, width=5, stride=2, padding="VALID")
        return conv(result, 18, 10, width=12, padding="VALID")


class simple3:

    def convolve(self, image, training,  keep_prob):
        result = conv_relu(image, 1, 18, width=5, stride=2, padding="VALID")
        result = conv(result, 18, 10, width=11, padding="VALID")
        return avg_pool(result)


class simple4:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5, stride=2, padding="VALID")
        return conv(result, 18, 10, width=12, padding="VALID")


class simple5:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5, padding="VALID")
        result = max_pool(result)  # 12
        result = conv_relu(result, 18, 24, width=5, padding="VALID")
        result = max_pool(result)  # 4
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 24, 10, width=4, padding="VALID")

# Epoch   200 , Training Loss 0.036521 , Test Loss 0.0219 , Train/Test 1.665 , Percent Error = 0.70 ,  93 seconds


class simple6:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5, padding="VALID")
        result = max_pool(result)  # 12
        result = conv_relu(result, 18, 24, width=5, padding="VALID")
        result = max_pool(result)  # 4
        return drop_conv(keep_prob, result, 24, 10, width=4, padding="VALID")

# Epoch   200 , Training Loss 0.032660 , Test Loss 0.0391 , Train/Test 0.835 , Percent Error = 0.66 ,  93 seconds


class simple7:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = conv_relu(result, 18, 24, width=5)
        result = max_pool(result)  # 7
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")

# Epoch   200 , Training Loss 0.018317 , Test Loss 0.0321 , Train/Test 0.571 , Percent Error = 0.67 ,  137 seconds


class simple8:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = drop_conv(keep_prob, result, 18, 24, width=5)
        result = tf.nn.relu(result)
        result = max_pool(result)  # 7
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")
# Epoch 500 Training Loss 0.0460221105441   Percent correct =  0.9927   345 seconds


class simple9:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = resnet_block(result, 18, 3, training)
        result = conv_relu(result, 18, 24, width=5)
        result = max_pool(result)  # 7
        result = resnet_block(result, 24, 3, training)
        result = resnet_block(result, 24, 3, training)
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")
# Epoch 400 Training Loss 7.6104698062e-05  Percent correct =  0.9936   640 seconds


class simple10:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = resnet_block(result, 18, 3, training)
        result = conv_relu(result, 18, 24, width=3)
        result = max_pool(result)  # 7
        result = resnet_block(result, 24, 3, training)
        result = resnet_block(result, 24, 3, training)
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")
# Epoch 800 Training Loss 2.16272041939e-05   Percent correct =  0.9948   1188 seconds


class simple11:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = resnet_block(result, 18, 3, training)
        result = resnet_block(result, 18, 3, training)
        result = conv_relu(result, 18, 24, width=3)
        result = max_pool(result)  # 7
        result = resnet_block(result, 24, 3, training)
        result = resnet_block(result, 24, 3, training)
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")
# Epoch 800 Training Loss 3.283573418e-06   Percent correct =  0.9943   1565 seconds


class simple12:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5, padding="VALID")
        result = max_pool(result)  # 12
        result = resnet_block(result, 18, 3, training)
        result = resnet_block(result, 18, 3, training)
        result = max_pool(result)  # 6
        result = conv_relu(result, 18, 24, width=1)
        result = resnet_narrow(result, 24, 3, training)
        result = resnet_narrow(result, 24, 3, training)
        result = max_pool(result)  # 3
        result = conv_relu(result, 24, 32, width=1)
        result = resnet_narrow(result, 32, 3, training)
        result = resnet_narrow(result, 32, 3, training)
        return drop_conv(keep_prob, result, 32, 10, width=3, padding="VALID")
# Epoch 800 Training Loss 0.00631735111412  Percent correct =  0.9868   1315 seconds


class simple13:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=3)
        result = max_pool(result)  # 14
        result = conv_relu(result, 18, 24, width=3)
        result = max_pool(result)  # 7
        return drop_conv(keep_prob, result, 24, 10, width=7, padding="VALID")
# Epoch 800 Training Loss 0.0190584493312   Percent correct =  0.9926   442 seconds


class simple14:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = tf.nn.relu(drop_conv(keep_prob, result, 18, 24, width=5))
        result = max_pool(result)  # 7
        result = tf.nn.relu(drop_conv(keep_prob, result, 24, 32, width=5, padding="VALID"))
        return conv(result, 32, 10, width=3, padding="VALID")


class simple15:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = conv_relu(result, 18, 24, width=5)
        result = max_pool(result)  # 7
        result = conv_relu(result, 24, 32, width=5, padding="VALID")
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 32, 10, width=3, padding="VALID")


class simple16:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = max_pool(result)  # 14
        result = conv_relu(result, 18, 24, width=5)
        result = max_pool(result)  # 7
        result = conv_relu(result, 24, 32, width=5)
        result = max_pool(result)  # 4
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 32, 10, width=4, padding="VALID")


class simple17:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = resnet_narrow(result, 18, 3, training)
        result = max_pool(result)  # 14
        result = resnet_narrow(result, 18, 3, training)
        result = conv_relu(result, 18, 24, width=5)
        result = resnet_narrow(result, 24, 3, training)
        result = max_pool(result)  # 7
        result = resnet_narrow(result, 24, 3, training)
        result = conv_relu(result, 24, 32, width=5, padding="VALID")
        result = resnet_narrow(result, 32, 3, training)
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 32, 10, width=3, padding="VALID")


class simple18:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = resnet_block(result, 18, 3, training, momentum=0.9)
        result = max_pool(result)  # 14
        result = resnet_block(result, 18, 3, training, momentum=0.9)
        result = conv_relu(result, 18, 24, width=5)
        result = resnet_block(result, 24, 3, training, momentum=0.9)
        result = max_pool(result)  # 7
        result = resnet_block(result, 24, 3, training, momentum=0.9)
        result = conv_relu(result, 24, 32, width=5, padding="VALID")
        result = resnet_block(result, 32, 3, training, momentum=0.9)
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 32, 10, width=3, padding="VALID")


class simple19:

    def convolve(self, image, training,  keep_prob):
        result = batch_normalization(image, training)
        result = conv_relu(result, 1, 18, width=5)
        result = resnet_block(result, 18, 3, training, momentum=0.99)
        result = max_pool(result)  # 14
        result = resnet_block(result, 18, 3, training, momentum=0.99)
        result = conv_relu(result, 18, 24, width=5)
        result = resnet_block(result, 24, 3, training, momentum=0.99)
        result = max_pool(result)  # 7
        result = resnet_block(result, 24, 3, training, momentum=0.99)
        result = conv_relu(result, 24, 32, width=5, padding="VALID")
        result = resnet_block(result, 32, 3, training, momentum=0.99)
        result = tf.nn.dropout(result, keep_prob)
        return conv(result, 32, 10, width=3, padding="VALID")


class conv_pool_with_msra:

    def convolve(self, image, training, keep_prob):
        layers = [1, 32, 64]
        width = 28
        conv_window = 3
        feature_layer_size = 128  # maybe 1024
        result = image

        for index in range(len(layers) - 1):
            result = conv_relu(result, layers[index], layers[index + 1], conv_window)
            result = resnet_block(result, layers=layers[index + 1], width=conv_window, training=training)
            result = resnet_block(result, layers=layers[index + 1], width=conv_window, training=training)
            result = max_pool(result)
            width = int(round(width / 2.0))

        result = conv_relu(result, layers[-1], feature_layer_size, width=width, padding='VALID')

        h_out = tf.reshape(result, [-1, feature_layer_size])
        h_out_drop = tf.nn.dropout(h_out, keep_prob)
        y = fully_connected(h_out_drop, feature_layer_size, 10)

        return y
