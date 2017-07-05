from __future__ import absolute_import
import tensorflow as tf
import time

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)


def weight_variable(shape):
  return tf.Variable( tf.truncated_normal(shape, stddev=0.1) )


def bias_variable(shape):
  return tf.Variable( tf.constant(0.1, shape=shape) )


def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer( x , layers_in , layers_out ):
  strides = [1, 1, 1, 1]
  w = weight_variable( [6, 6, layers_in, layers_out] ) 
  b = bias_variable( [layers_out] ) 
  h = tf.nn.conv2d( x, w, strides=strides, padding='VALID' ) + b
  return tf.nn.relu( h )


def model_just_fully_connected() :
  # inputs
  x  = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  # unused for this model
  keep_prob = tf.placeholder(tf.float32)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x,y_,keep_prob,train,percent_correct


def model_conv_pool_x2() :
  # inputs
  x  = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  train = tf.train.AdamOptimizer(1e-4).minimize(loss)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x,y_,keep_prob,train,percent_correct


def model_conv(levels) :
  # inputs
  x  = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv = x_image
  layers = 1
  conv_size = 28
  for _ in range(levels) :
    next_layers = max(24,layers + 2)
    h_conv = conv_layer( h_conv , layers , next_layers )
    layers = next_layers
    conv_size -= 5

  conv_size = conv_size**2 * layers

  h1 = tf.reshape(h_conv, [-1, conv_size ])

  W1 = weight_variable([ conv_size , 128])
  b1 = bias_variable([128])
  h1 = tf.nn.relu(tf.matmul(h1, W1) + b1)

  keep_prob = tf.placeholder(tf.float32)
  h1_drop = tf.nn.dropout(h1, keep_prob)

  W2 = weight_variable([128, 10])
  b2 = bias_variable([10])
  y = tf.matmul(h1_drop, W2) + b2

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  train = tf.train.AdamOptimizer(1e-2).minimize(loss)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x,y_,keep_prob,train,percent_correct




def train_model_and_report(mnist,model):

  x,y_,keep_prob,train,percent_correct = model

  with tf.Session() as sess :
    sess.run( tf.global_variables_initializer() )

    max_count = 10000
    count = 0
    report_round = 0
    while report_round < 5 and count < max_count:
      report_round += 1
      end = time.time() + 60
      while time.time() < end and count < max_count:
        count += 1
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  
      print "\tTraining round",report_round,"complete. ",count," training batches."
      print "\tPercent correct =",sess.run(percent_correct, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})




print """
Mnist model comparison.

This program trains several different tensorflow models on the
Mnist dataset.  Each model runs for a maximum of 10,000 training
batches or 5 minutes, whichever comes first.  The training is
broken into 5 reportings of 1 minute of run time each.

Total run time will be approximately 30 minutes after loading
the Mnist data.

Downloading and opening Mnist data...
"""

mnist = get_mnist_data()
"""
print "\nTraining model_just_fully_connected ..."
train_model_and_report(mnist,model_just_fully_connected())

print "\nTraining model_conv_pool_x2 ..."
train_model_and_report(mnist,model_conv_pool_x2())

print "\nTraining model_conv_x1 ..."
train_model_and_report(mnist,model_conv(1))

print "\nTraining model_conv_x2 ..."
train_model_and_report(mnist,model_conv(2))

print "\nTraining model_conv_x3 ..."
train_model_and_report(mnist,model_conv(3))

print "\nTraining model_conv_x4 ..."
train_model_and_report(mnist,model_conv(4))
"""
print """
Complete.
"""


