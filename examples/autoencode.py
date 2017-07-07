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
stages.append( layer.conv_relu( stages[-1], 18, 20, width=5, padding="VALID" ) )
stages.append( layer.conv_relu( stages[-1], 20, 24, width=3, padding="VALID", name="embedding") )
stages.append( layer.relu_deconv( stages[-1], 24, 20, width=3, shape=tf.shape(stages[2]) ) )
stages.append( layer.relu_deconv( stages[-1], 20, 18, width=5, shape=tf.shape(stages[1]) ) )
stages.append( tf.nn.relu( layer.relu_deconv( stages[-1], 18, 1, width=8, shape=tf.shape(stages[0]) ), name="x_out" ) ) 

x_in = stages[0]
x_out = stages[-1]

encode_loss = tf.reduce_mean( tf.reduce_mean( tf.square( x_in - x_out ) ) )
encode_train = tf.train.AdamOptimizer(1e-3).minimize(encode_loss)

y0 = tf.placeholder(tf.float32, [None, 10])
embedding = tf.stop_gradient( tf.reshape( stages[3] , [tf.shape(stages[3])[0],24] ) )
y_logit = layer.fully_connected( embedding , 24 , 10 )
y_out = tf.nn.softmax( y_logit )

classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y0))
classify_train = tf.train.AdamOptimizer(1e-4).minimize(classify_loss)


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

def doAutoEncodeTraining( batches, batch_size=200 ) :
  print "doAutoEncodeTraining"
  for index in range(1,batches+1) :
    result,_ = sess.run( [encode_loss,encode_train], feed_dict={x0:mnist.train.next_batch(batch_size)[0]})
    if index == 1 or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        print "index :",index,", loss :",result

doAutoEncodeTraining(1000)

def doClassifyTraining( batches, batch_size=200 ) :
  print "doClassifyTraining"
  for index in range(1,batches+1) :
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    result,_ = sess.run( [classify_loss,classify_train], feed_dict={x0:batch_x,y0:batch_y})
    if index == 1 or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        print "index :",index,", loss :",result

doClassifyTraining(1000)



def showExample() :
    sample_x,sample_y = mnist.train.next_batch(1)
    print "The number is",np.argmax(sample_y[0])
    sample_in,sample_out = sess.run([x_in,x_out],feed_dict={x0:sample_x} )
    print "===== IN ====="
    print (sample_in.reshape([14,14]) * 5 ).round().astype(int)
    print "===== OUT ====="
    print (sample_out.reshape([14,14]) * 5 ).round().astype(int)

for _ in range(5):
    showExample()
