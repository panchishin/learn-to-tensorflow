import tensorflow as tf
import numpy as np
import layer


SIZE = 28

learning_rate = tf.placeholder( dtype=tf.float32 )
x0 = tf.placeholder(tf.float32, [None, SIZE*SIZE])
x_reshape = tf.reshape( x0, [-1,SIZE,SIZE,1] ) 

stages = []
stages.append( layer.avg_pool( x_reshape, name="x_in" ) )
stages.append( layer.conv_relu( stages[-1], 1, 18, width=8, padding="VALID" ) )
stages.append( layer.conv_relu( stages[-1], 18, 24, width=5, padding="VALID" ) )
stages.append( layer.conv_relu( stages[-1], 24, 32, width=3, padding="VALID", name="embedding") )
stages.append( layer.relu_deconv( stages[-1], 32, 24, width=3, shape=tf.shape(stages[2]) ) )
stages.append( layer.relu_deconv( stages[-1], 24, 18, width=5, shape=tf.shape(stages[1]) ) )
stages.append( layer.relu_deconv( stages[-1], 18, 1, width=8, shape=tf.shape(stages[0]), name="x_out" ) )

x_in = stages[0]
x_out = tf.nn.relu(stages[-1])
embedding = stages[3]

loss = tf.reduce_mean( tf.reduce_mean( tf.square( x_in - x_out ) ) ) 
loss = loss + tf.reduce_mean( tf.reduce_mean( tf.square( stages[1] - stages[-2] ) ) ) 
loss = loss + tf.reduce_mean( tf.reduce_mean( tf.square( stages[2] - stages[-3] ) ) ) 

train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()


sess = tf.Session()
sess.run( tf.global_variables_initializer() )

print """
==== Tensor Shapes ====
"""
results = sess.run(stages, feed_dict={x0:mnist.train.next_batch(4)[0]})
for result,stage in zip(results,stages) :
    print result.shape , stage.name
print """
=======================
"""

def doTraining( amount , learning_rate_value ) :
  for index in range(1,amount+1) :
    result,_ = sess.run( [loss,train], feed_dict={x0:mnist.train.next_batch(100)[0],learning_rate:learning_rate_value})
    if index == 1 or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        print "index :",index,", loss :",result

doTraining(20000,1e-3)

def showExample() :
    sample_x,sample_y = mnist.train.next_batch(1)
    print "The number is",np.argmax(sample_y[0])
    sample_in,sample_out = sess.run([x_in,x_out],feed_dict={x0:sample_x} )
    print "===== IN ====="
    print (sample_in.reshape([14,14]) * 5 ).round().astype(int)
    print "===== OUT ====="
    print (sample_out.reshape([14,14]) * 5 ).round().astype(int)

showExample()
