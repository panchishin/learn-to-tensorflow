### imports
import tensorflow as tf
import read_file_helper as helper
import time
import numpy as np
import os
import random

file_name = "temp.save"

text_length = 50
helper.set("text-input-file.txt",text_length)

labels = 256
neurons = labels/2
layers = 3
steps_per_report = 500

print "text_length",text_length,", labels",labels, ", neurons",neurons, ", layers",layers, ", steps_per_report",steps_per_report

### induction
# input layer
learning_rate = tf.placeholder( dtype=tf.float32 )

x0 = tf.placeholder( dtype=tf.int32, shape=[None] , name="x0")
y0 = tf.placeholder( dtype=tf.int32, shape=[None] , name="y0")

# one-hot encoding
x0_hot = tf.one_hot( x0, labels, dtype=tf.float32 , name="x0_hot" )
x0_rnn_shape = tf.reshape( x0_hot , [1,-1,labels] , name="x0_rnn_shape" )
y0_hot = tf.one_hot( y0, labels, dtype=tf.float32 , name="y0_hot" )

# RNN layer cell definition
cell = tf.nn.rnn_cell.GRUCell( neurons )
cell = tf.nn.rnn_cell.MultiRNNCell( [cell] * layers )

cell_zero_state = cell.zero_state( 1 , tf.float32  )
init_state = tf.Variable( cell_zero_state , trainable=False , name="init_state" )
set_init_state = tf.assign( init_state,  cell_zero_state )

dynamic_rnn_init_state = tf.split( 0, layers, tf.reshape( init_state, [layers,neurons] ) )
output, state = tf.nn.dynamic_rnn( cell, x0_rnn_shape, initial_state = tuple(dynamic_rnn_init_state) )
last_output = tf.reshape(output,[-1,neurons])

# fully connected output layer
m = tf.Variable( tf.random_normal( dtype=tf.float32, shape=[neurons,labels] ), name="m" )
b = tf.Variable( tf.random_normal( dtype=tf.float32, shape=[labels] ), name="b" )
z = tf.matmul( last_output , m ) + b

y_out = tf.nn.softmax( z )
y_argmax = tf.argmax(y_out,1)

### loss
# use basic cross entropy
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z,y0_hot) )

### training
# use adam or gradient decent optimizer with 0.003
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
update_state = tf.assign( init_state , state )


learning_rate_value = 0.0015

### Execution
print "Starting execution with learning rate",learning_rate_value
with tf.Session() as sess:

  saver = tf.train.Saver()
  if os.path.exists(file_name) :
    print "Restoring state from previous save..."
    saver.restore(sess,file_name)
  else :
    print "Initializing state..."
    # initialize session variables
    sess.run( tf.initialize_all_variables() )
    sess.run( set_init_state )
  print "...done"

  previousLoss = 10.0

  for report in range(200):
    start_time = time.time()
    # initialize avgloss and num correct to 0 and 0.0 respectively
    avgloss, epoch = 0, 0.0

    for step in range(steps_per_report) :
      # retrieve x and y example data using makeXY()
      inx,outy,epoch = helper.getXY( report*steps_per_report + step )

      # execute 'loss' and 'y_argmax' to gather stats
      junk_a,thisloss,result,junk_b = sess.run( [train,loss,y_argmax,update_state] , feed_dict={x0:inx,y0:outy,learning_rate:learning_rate_value})

      # add thisloss to avgloss
      avgloss += thisloss

    avgloss = avgloss / steps_per_report
    learning_rate_value *= 1.01 if avgloss < previousLoss else 0.9
    previousLoss = previousLoss*0.9 + avgloss*0.1
    print "Epoch =",epoch," , avgloss =",avgloss,", learning rate",learning_rate_value


    for temperature in [0.5,0.8] :
      print "sample output for temperature",temperature
      sentence = [ ord(" ") ]
      for _ in range(40) :
        lastProbs = np.array( sess.run(y_out, feed_dict={x0:sentence}) )[-1,:]
        lastProbsIndexDescOrder = sorted(range(len(lastProbs)), key=lambda k: -lastProbs[k])

        index = 0
        random = np.random.random()*temperature
        while random > lastProbs[lastProbsIndexDescOrder[index]] :
          random -= lastProbs[lastProbsIndexDescOrder[index]]
          index += 1
        sentence.append(lastProbsIndexDescOrder[index])

      sentence.append(ord("."))
      print "".join([ chr(c) for c in sentence ])

    print " "
    print "Elapse =",(time.time() - start_time),"seconds,",((steps_per_report*text_length)/(time.time() - start_time)),"characters per second"
    print " "
    print "saving..."
    start_time = time.time()
    saver.save(sess,file_name)
    print "...done in",(time.time() - start_time),"seconds"
    
