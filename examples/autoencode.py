import tensorflow as tf
import numpy as np
import layer
import sys
print sys.argv


SIZE = 28
HIGH_LOW_NOISE = .02
EMBED_SIZE = 28

learning_rate = tf.placeholder( dtype=tf.float32 )
x0 = tf.placeholder(tf.float32, [None, SIZE*SIZE])

stages = []
stages.append( tf.reshape( x0, [-1,SIZE,SIZE,1], name="x_in" )  )
# convolution starts
stages.append( layer.conv_relu( layer.high_low_noise( stages[-1] , HIGH_LOW_NOISE), 1, 18, width=5, padding="VALID" ) )
stages.append( layer.max_pool( stages[-1] ) )
stages.append( layer.conv_relu( stages[-1], 18, 24, width=5, padding="VALID" ) )
stages.append( layer.max_pool( stages[-1] ) )
stages.append( layer.conv_relu( stages[-1], 24, EMBED_SIZE, width=4, padding="VALID", name="embedding" ) )
# embedding
embedding = tf.reshape( stages[-1] , [tf.shape(stages[-1])[0],EMBED_SIZE] )
# deconvolution starts

if sys.argv[1] == "conv" :
  print "=== adding extra upscale and convolution ==="
  stages.append( layer.upscaleFlat( stages[-1] , scale=4 ) )
  stages.append( layer.conv_relu( stages[-1], EMBED_SIZE, 24, width=4, padding="SAME" ) )
  stages.append( layer.upscaleFlat( stages[-1] , scale=4 ) )
  stages.append( layer.conv_relu( stages[-1], 24, 18, width=5, padding="SAME" ) )
  stages.append( layer.upscaleFlat( stages[-1] , scale=2 ) )
  stages.append( layer.conv_relu( stages[-1], 18, 1, width=5, padding="VALID" ) )

if sys.argv[1] == "deconv" :
  print "=== using deconvolution ==="
  stages.append( layer.relu_deconv( stages[-1], EMBED_SIZE, 24, width=4, shape=tf.shape(stages[4]) ) ) 
  stages.append( layer.upscaleFlat( stages[-1] , scale=2 ) )
  stages.append( layer.relu_deconv( stages[-1], 24, 18, width=5, shape=tf.shape(stages[2]) ) ) 
  stages.append( layer.upscaleFlat( stages[-1] , scale=2 ) )
  stages.append( layer.relu_deconv( stages[-1], 18, 1, width=5, shape=tf.shape(stages[0]) ) ) 


stages.append( tf.nn.relu( stages[-1] , name="x_out" ) )

x_reshape = stages[0]
x_in = stages[0]
x_out = stages[-1]

encode_loss = tf.reduce_mean( tf.reduce_mean( tf.square( x_in - x_out ) ) )
encode_train = tf.train.AdamOptimizer(1e-3).minimize(encode_loss)

y0 = tf.placeholder(tf.float32, [None, 10])
out_fc_1 = tf.nn.relu( layer.fully_connected( embedding , EMBED_SIZE , EMBED_SIZE ) )
y_logit = layer.fully_connected( out_fc_1 , EMBED_SIZE , 10 )
y_out = tf.nn.softmax( y_logit )

classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y0))
classify_train = tf.train.AdamOptimizer(1e-3).minimize(classify_loss)

total_loss = tf.reduce_sum( tf.log( tf.stack( [encode_loss,classify_loss] ) ) )
total_train = tf.train.AdamOptimizer(1e-4).minimize(total_loss)


y_out_index = tf.argmax(y_out,1)
y_in_index = tf.argmax(y0,1)
correct = tf.reduce_mean( tf.cast( tf.equal(y_out_index,y_in_index) , tf.float32 ) )


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
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    batch_correct,result,_ = sess.run( [correct,encode_loss,encode_train], feed_dict={x0:batch_x,y0:batch_y})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={x0:mnist.test.images,y0:mnist.test.labels})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )

def doClassifyTraining( batches, batch_size=200 ) :
  print "doClassifyTraining"
  for index in range(1,batches+1) :
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    batch_correct,result,_ = sess.run( [correct,classify_loss,classify_train], feed_dict={x0:batch_x,y0:batch_y})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={x0:mnist.test.images,y0:mnist.test.labels})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )

def doTotalTraining( batches, batch_size=200 ) :
  print "doTotalTraining"
  for index in range(1,batches+1) :
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    batch_correct,result,_ = sess.run( [correct,total_loss,total_train], feed_dict={x0:batch_x,y0:batch_y})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={x0:mnist.test.images,y0:mnist.test.labels})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )


def showExample(count) :
  for _ in range(count) :
    sample_x,sample_y_in = mnist.train.next_batch(1)
    sample_in,sample_out,sample_y_out = sess.run([x_in,x_out,y_out],feed_dict={x0:sample_x} )
    print"Estimated / Actual class [chance] : ",
    print np.argmax(sample_y_out[0]),"[",int(round(100.0*sample_y_out[0][np.argmax(sample_y_out[0])])),"]",
    print "/",
    print np.argmax(sample_y_in[0]),"[",int(round(100.0*sample_y_out[0][np.argmax(sample_y_in[0])])),"]",
    print "===== IN ====="
    print (sample_in.reshape([28,28]) * 5 ).round().astype(int)
    print "===== OUT ====="
    print (sample_out.reshape([28,28]) * 5 ).round().astype(int)

def confusion() :
  y_in_result,y_out_result = sess.run( [y_in_index,y_out_index], feed_dict={x0:mnist.test.images,y0:mnist.test.labels})
  for in_index in range(10) :
    print in_index," = ",
    for out_index in range(10) :
      mask =  (y_in_result==in_index)*(y_out_result==out_index)
      print "%4.0f " % mask.sum(),
    print " "



def error_result() :
  correct_result = sess.run( correct, feed_dict={x0:mnist.test.images,y0:mnist.test.labels})
  return round(100*(1 - correct_result))


def help() :
  print """
showExample(2)
doAutoEncodeTraining(10000)
doClassifyTraining(10000)
doTotalTraining(10000)
confusion()
error_result()
"""

help()


doTotalTraining(30000)
