### imports
import tensorflow as tf
import numpy as np

labels = 10
neurons = 20
layers = 2
steps_per_epoch = 500

### induction

# input layer
dropout = tf.placeholder( dtype=tf.float32 )
x0 = tf.placeholder( dtype=tf.int32, shape=[None,None])
y0 = tf.placeholder( dtype=tf.int32, shape=[None] )

# one-hot encoding
x0_hot = tf.one_hot( x0, labels, dtype=tf.float32 )
y0_hot = tf.one_hot( y0, labels, dtype=tf.float32 )

# RNN layer cell definition
cell = tf.nn.rnn_cell.GRUCell( neurons )
cell = tf.nn.rnn_cell.DropoutWrapper( cell, output_keep_prob=dropout )
cell = tf.nn.rnn_cell.MultiRNNCell( [cell] * layers )

# RNN output
output, state = tf.nn.dynamic_rnn(cell, x0_hot, dtype=tf.float32)
last_output = tf.reshape(output[-1,-1],[-1,neurons])

# fully connected output layer
m = tf.Variable( tf.random_normal( dtype=tf.float32, shape=[neurons,labels] ) )
b = tf.Variable( tf.random_normal( dtype=tf.float32, shape=[labels] ) )
z = tf.matmul( last_output , m ) + b
y_out = tf.nn.softmax( tf.reshape( z , [labels] ) )
y_argmax = tf.argmax(y_out,0)


### loss
# use basic cross entropy
loss = -tf.reduce_mean(tf.reduce_sum( y0_hot * tf.log(y_out) ))


### training
# use adam or gradient decent optimizer with 0.01 
train = tf.train.AdamOptimizer(0.01).minimize(loss)


### Creation of series data.
# The X is a series like [2,3,4] and the expected result is [5]
# Also makes decreasing and constant series
# of length 4,3,and 2
def makeXY() :
  x,y = makeXYfirst()
  if np.random.random() > 0.66 :
    x = x[1:]
  elif np.random.random() > 0.5 :
    x = x[2:]
  return x,y 

def makeXYfirst() :
  if np.random.random() > 0.66 :
    x = int( np.random.random() * 6. ) + np.array([0,1,2,3])
    y = x[-1] + 1
    return x,y
  if np.random.random() > 0.5 :
    x = int( np.random.random() * 6. ) + np.array([4,3,2,1])
    y = x[-1] - 1
    return x,y
  else : 
    x = int( np.random.random() * 10. ) + np.array([0,0,0,0])
    y = x[-1] 
    return x,y


### Execution
with tf.Session() as sess:
  # initialize session variables
  sess.run( tf.initialize_all_variables() )

  for epoch in range(20):
    # initialize avgloss and num correct to 0 and 0.0 respectively
    avgloss, correct = 0, 0.0

    for step in range(steps_per_epoch) :
      # retrieve x and y example data using makeXY()
      inx,outy = makeXY()

      # execute 'train' with dropout of 0.5 to train a resilient NN
      sess.run( train , feed_dict={x0:[inx],y0:[outy],dropout:0.5})

      # execute 'loss' and 'y_argmax' with dropout of 1.0 to gather stats
      thisloss,result = sess.run( [loss,y_argmax] , feed_dict={x0:[inx],y0:[outy],dropout:1.0})
      # add thisloss to avgloss
      avgloss += thisloss
      # increment correct if result is the same as outy
      correct += 1.0 if result == outy else 0.0

    print "Epoch =",epoch," , correct =",(correct/steps_per_epoch)," , avgloss =",(avgloss / steps_per_epoch)
    if correct == steps_per_epoch :
      print "Finished"
      break

