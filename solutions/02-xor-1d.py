## Lesson 2 - solve XOR
# use random variables in the range of 0.1 to 0.9 and 32 bit floating precision

### imports
import tensorflow as tf

### constant data
x = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y = [[0.],[0.],[1.],[1.]]

### induction
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = the x2 inputs
x0 = tf.constant( x , dtype=tf.float32 )
y0 = tf.constant( y , dtype=tf.float32 )

# Layer 1 = the 2x3 hidden sigmoid
m1 = tf.Variable( tf.random_uniform( [2,3] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b1 = tf.Variable( tf.random_uniform( [3] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
h1 = tf.sigmoid( tf.matmul( x0,m1 ) + b1 )

# Layer 2 = the 3x1 sigmoid output
m2 = tf.Variable( tf.random_uniform( [3,1] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b2 = tf.Variable( tf.random_uniform( [1] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
y_ = tf.sigmoid( tf.matmul( h1,m2 ) + b2 )


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
      sess.run(train)
    print sess.run(loss)

  results = sess.run([m1,b1,m2,b2,y_,loss])
  labels  = "m1,b1,m2,b2,y_,loss".split(",")
  for label,result in zip(*(labels,results)) :
    print ""
    print label
    print result

print ""
