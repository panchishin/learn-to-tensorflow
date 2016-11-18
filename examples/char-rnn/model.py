### imports
import tensorflow as tf

labels = 256
neurons = labels/2
layers = 3

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
cell = tf.nn.rnn_cell.LSTMCell( neurons )
cell = tf.nn.rnn_cell.MultiRNNCell( [cell] * layers )

cell_zero_state = cell.zero_state( 1 , tf.float32  )
init_state = tf.Variable( cell_zero_state , trainable=False , name="init_state" )
set_init_state = tf.assign( init_state,  cell_zero_state )

init_state_for_rnn1 = tf.split( 0, layers*2, tf.reshape( init_state, [layers*2,neurons] ) )
init_state_for_rnn2 = tuple( [ tf.nn.rnn_cell.LSTMStateTuple(init_state_for_rnn1[x*2],init_state_for_rnn1[x*2+1]) for x in range(layers) ] )

output, state = tf.nn.dynamic_rnn( cell, x0_rnn_shape, initial_state=init_state_for_rnn2 )
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


import numpy as np
def randomLabel(output_probs , temp=0.7) :
  ordered_probs = sorted(range(len(output_probs)), key=lambda k: -output_probs[k])
  index = 0
  random = np.random.random()*temp
  while random > output_probs[ordered_probs[index]] :
    random -= output_probs[ordered_probs[index]]
    index += 1
  return ordered_probs[index]
