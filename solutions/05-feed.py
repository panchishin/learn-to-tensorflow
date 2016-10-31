## Lesson 4 - XOR with 2 outputs
# Change lesson 2 to have two softmax output channels instead of sigmoid

### imports
import tensorflow as tf

### constant data
x = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]

### induction
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = the x2 inputs
x0 = tf.placeholder( dtype=tf.float32 , shape=[None,2] )
y0 = tf.placeholder( dtype=tf.float32 , shape=[None,2] )

# Layer 1 = the 2x3 hidden sigmoid
m1 = tf.Variable( tf.random_uniform( [2,3] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b1 = tf.Variable( tf.random_uniform( [3] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
h1 = tf.sigmoid( tf.matmul( x0,m1 ) + b1 )

# Layer 2 = the 3x2 softmax output
m2 = tf.Variable( tf.random_uniform( [3,2] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b2 = tf.Variable( tf.random_uniform( [2] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
y_ = tf.nn.softmax( tf.matmul( h1,m2 ) + b2 )


### loss

# loss : sum of the squares of y0 - y_
loss = tf.reduce_sum( tf.square( y0 - y_ ) )

# training step : gradient decent (1.0) to minimize loss
train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)


### training
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
  sess.run( tf.initialize_all_variables() )

  print "\nloss"
  for epoc in range(5):
    for step in range(100) :
      sess.run(train, feed_dict={x0:x,y0:y})
    print sess.run(loss, feed_dict={x0:x,y0:y})

  results = sess.run([m1,b1,m2,b2,y_,loss], feed_dict={x0:x,y0:y})
  labels  = "m1,b1,m2,b2,y_,loss".split(",")
  for label,result in zip(*(labels,results)) :
    print ""
    print label
    print result

print ""
