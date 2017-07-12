import tensorflow as tf
import numpy as np
import layer
import sys
print sys.argv


SIZE = 28
HIGH_LOW_NOISE = .02
EMBED_SIZE = 28

learning_rate = tf.placeholder( dtype=tf.float32, name="learning_rate" )
training_flag = tf.Variable( tf.constant(True), name="training_flag")

x0 = tf.placeholder(tf.float32, [None, SIZE*SIZE] , name="x0")
x_reshape = tf.reshape( x0, [-1,SIZE,SIZE,1], name="x_in" )
x_noisy = layer.high_low_noise( x_reshape , HIGH_LOW_NOISE)


def convolution_layers(x_in, layer_sizes=[1,18,24,28], width_sizes=[5,5,4], training=True, resnets=0) :
  stages = [x_in]
  for in_size,out_size,width,end in zip(layer_sizes[:-1],layer_sizes[1:],width_sizes,range(len(width_sizes)-1,-1,-1)) :
    stages.append( layer.conv_relu( stages[-1], in_size, out_size, width=width, padding="VALID" ) )
    for _ in range(resnets) :
      stages.append( layer.resnet_block( stages[-1], out_size, 3 , training ) )
    if end != 0 :
      stages.append( layer.max_pool( stages[-1] ) )
  return stages

def deconvolution_layers(stages, layer_sizes=[1,18,24,28], width_sizes=[5,5,4], training=True, resnets=0) :
  for in_size,out_size,width,end in zip(layer_sizes[:-1],layer_sizes[1:],width_sizes,range(len(width_sizes)))[::-1] :
    for _ in range(resnets) :
      stages.append( layer.resnet_block( stages[-1], out_size, 3 , training ) )
    stages.append( layer.relu_deconv( stages[-1], out_size, in_size, width=width, shape=tf.shape(stages[end*(2+resnets)]) ) ) 
    if end != 0 :
      stages.append( layer.upscaleFlat( stages[-1] , scale=2 ) )
  return stages

def autoencode(x_in, layer_sizes=[1,18,24,28], width_sizes=[5,5,4], training=True, resnets=1) :
  print "==== Convolution ===="
  stages = convolution_layers(x_in, layer_sizes, width_sizes, training=training, resnets=resnets)
  print "==== Embedding ===="
  embedding = stages[-1]
  print "==== Deconvolution ===="
  return embedding, deconvolution_layers(stages, layer_sizes, width_sizes, training=training, resnets=resnets)


embedding, stages = autoencode( x_noisy , layer_sizes=[1,18,24,EMBED_SIZE], width_sizes=[5,5,4], training=training_flag, resnets=1)


x_in = x_reshape
x_out = tf.nn.relu( stages[-1] , name="x_out" )

encode_loss = tf.reduce_mean( tf.reduce_mean( tf.square( x_in - x_out ) ) )
encode_train = tf.train.AdamOptimizer(1e-3).minimize(encode_loss)


y0 = tf.placeholder(tf.float32, [None, 10], name="y0")
freeze_embedding = tf.placeholder(tf.bool, name="freeze_embedding")
embedding_flat = tf.reshape( embedding , [tf.shape(stages[-1])[0],EMBED_SIZE] )
filter_embedding = tf.cond( freeze_embedding , lambda: tf.stop_gradient(embedding_flat) , lambda: tf.identity(embedding_flat) )
out_fc_1 = tf.nn.relu( layer.fully_connected( filter_embedding , EMBED_SIZE , EMBED_SIZE ) )
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
    batch_correct,result,_ = sess.run( [correct,encode_loss,encode_train], feed_dict={x0:batch_x,y0:batch_y,freeze_embedding:True})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={training_flag:False,x0:mnist.test.images,y0:mnist.test.labels,freeze_embedding:True})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )

def doClassifyTraining( batches, batch_size=200, freeze=True ) :
  print "doClassifyTraining"
  for index in range(1,batches+1) :
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    batch_correct,result,_ = sess.run( [correct,classify_loss,classify_train], feed_dict={x0:batch_x,y0:batch_y,freeze_embedding:freeze})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={training_flag:False,x0:mnist.test.images,y0:mnist.test.labels,freeze_embedding:freeze})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )

def doTotalTraining( batches, batch_size=200, freeze=True ) :
  print "doTotalTraining"
  for index in range(1,batches+1) :
    batch_x , batch_y = mnist.train.next_batch(batch_size)
    batch_correct,result,_ = sess.run( [correct,total_loss,total_train], feed_dict={x0:batch_x,y0:batch_y,freeze_embedding:freeze})
    if index == 1 or ( index < 100 and index % 10 == 0 ) or ( index < 1000 and index % 100 == 0 ) or index % 1000 == 0 :
        correct_result = sess.run( correct, feed_dict={training_flag:False,x0:mnist.test.images,y0:mnist.test.labels,freeze_embedding:freeze})
        print "index : %5d, loss: %5.3f, train error %4.1f, test error %4.1f percent" % ( index, result, 100*(1 - batch_correct), 100*(1 - correct_result) )


def showExample(count) :
  for _ in range(count) :
    sample_x,sample_y_in = mnist.train.next_batch(1)
    sample_in,sample_out,sample_y_out = sess.run([x_in,x_out,y_out],feed_dict={training_flag:False,x0:sample_x,freeze_embedding:True} )
    print"Estimated / Actual class [chance] : ",
    print np.argmax(sample_y_out[0]),"[",int(round(100.0*sample_y_out[0][np.argmax(sample_y_out[0])])),"]",
    print "/",
    print np.argmax(sample_y_in[0]),"[",int(round(100.0*sample_y_out[0][np.argmax(sample_y_in[0])])),"]",
    print "===== IN ====="
    print (sample_in.reshape([28,28]) * 5 ).round().astype(int)
    print "===== OUT ====="
    print (sample_out.reshape([28,28]) * 5 ).round().astype(int)


def confusion() :
  y_in_result,y_out_result = sess.run( [y_in_index,y_out_index], feed_dict={training_flag:False,x0:mnist.test.images,y0:mnist.test.labels,freeze_embedding:True})
  for in_index in range(10) :
    print in_index," = ",
    for out_index in range(10) :
      mask =  (y_in_result==in_index)*(y_out_result==out_index)
      print "%4.0f " % mask.sum(),
    print " "



def error_result() :
  correct_result = sess.run( correct, feed_dict={training_flag:False,x0:mnist.test.images,y0:mnist.test.labels,freeze_embedding:True})
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
