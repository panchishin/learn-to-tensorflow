from __future__ import absolute_import
import tensorflow as tf
import time

def parameter_count( shape , name="") :
  print name," Parametes ",shape,", Count :",reduce(lambda x, y: x*y, shape )

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./mnist_data', one_hot=True)

def weight_variable(shape, name="Weight_Variable"):
  parameter_count(shape,name)
  return tf.Variable( tf.truncated_normal(shape, stddev=0.01), name=name )

def bias_variable(shape):
  return tf.Variable( tf.constant(0.1, shape=shape) )

def max_pool(x,stride=2):
  return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x,stride=2):
  return tf.nn.avg_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def conv( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv" ):
  w = weight_variable( [width, width, layers_in, layers_out] , name=name) 
  b = bias_variable( [layers_out] ) 
  return tf.nn.conv2d( x, w, strides= [1, stride, stride, 1], padding=padding ) + b

def drop_conv( keep_prob, x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="drop_conv" ):
  w = weight_variable( [width, width, layers_in, layers_out] , name=name ) 
  w = tf.nn.dropout( w, keep_prob ) / keep_prob
  b = bias_variable( [layers_out] ) 
  return tf.nn.conv2d( x, w, strides= [1, stride, stride, 1], padding=padding ) + b

def conv_relu( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv_relu" ):
  h = conv( x , layers_in , layers_out , width , stride, padding, name=name )
  return tf.nn.relu( h )

def batch_normalization( x, training, momentum=0.9 ) :
  return tf.layers.batch_normalization( x, training=training, momentum=momentum )

def single_resnet_block( x, layers, width , training, momentum=0.9, name="single_resnet_block" ) :
  result = batch_normalization( x , training, momentum=momentum )
  result = tf.nn.relu(result)
  return conv( result, layers, layers, width=width, name=name )

def resnet_block( x, layers, width , training, momentum=0.9, name="resnet_block" ) :
  result = single_resnet_block( x,      layers, width, training, momentum=momentum, name=name )
  result = single_resnet_block( result, layers, width, training, momentum=momentum, name=name )
  return tf.add( x , result )

def resnet_narrow( x, layers, width , training, narrowing=2, name="resnet_narrow" ) :
  result = batch_normalization( x , training )
  result = tf.nn.relu(result)
  result = conv( result, layers, layers/narrowing, width=1, name=name )
  result = tf.nn.relu(result)
  result = conv( result, layers/narrowing, layers/narrowing, width=width, name=name )
  result = tf.nn.relu(result)
  result = conv( result, layers/narrowing, layers, width=1, name=name )
  result = tf.nn.relu(result)
  return tf.add( x , result )

def fully_connected( x , size_in , size_out, name="fully_connected" ):
  W = weight_variable( [size_in, size_out], name=name )
  b = bias_variable( [size_out] )
  return tf.matmul(x, W) + b


def handler_wrapper( handler ) :
  # inputs
  tf.reset_default_graph()

  size         = 28
  x            = tf.placeholder(tf.float32, [None, size*size])
  y_           = tf.placeholder(tf.float32, [None, 10])
  keep_prob    = tf.placeholder(tf.float32)
  learning_rate = tf.placeholder(tf.float32)
  training     = tf.placeholder(tf.bool)
  h_pool       = tf.reshape(x, [-1,size,size,1])

  h_pool       = handler.convolve( h_pool, training,  keep_prob )
  y            = tf.reshape(h_pool, [-1,10])

  loss         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

  update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train        = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x,y_,keep_prob,loss,train,percent_correct,training,learning_rate


def train_model_and_report(mnist,model,learning_rate_value=1e-4,epochs=200,keep_prob_value=0.5):
  print "MODEL :",model.__class__.__name__
  x,y_,keep_prob,loss,train,percent_correct,training,learning_rate = handler_wrapper( model )

  with tf.Session() as sess :
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )

    items_per_batch = 200
    start_time = time.time()

    lambda_error = 100.0
    lambda_val   = 0.3

    for epoch in range(1,epochs+1) :
      if (epoch+1) % (epochs / 4) == 0 :
        learning_rate_value *= .5
        #print "Decreasing learning_rate to", learning_rate_value
        
      training_loss = 0.0
      for _ in range( mnist.test.labels.shape[0] / items_per_batch ) :
        batch_xs, batch_ys = mnist.train.next_batch(items_per_batch)
        result_loss,_ = sess.run([loss,train], feed_dict={
          x: batch_xs, y_: batch_ys, 
          keep_prob: keep_prob_value, training:True, learning_rate:learning_rate_value})
        training_loss += result_loss

      training_loss *= 1.0 * items_per_batch / mnist.test.labels.shape[0]

      if epoch % 5 == 0 :
        print "\tEpoch %5d"%epoch,", Training Loss %0.6f"%training_loss,
        correct = 0.0
        test_loss = 0.0
        for test_batch in range(100) :
          test_batch_start = test_batch * 100
          test_batch_end   = test_batch_start + 100
          result_loss,result_correct = sess.run([loss,percent_correct], feed_dict={
            x: mnist.test.images[test_batch_start:test_batch_end], 
            y_: mnist.test.labels[test_batch_start:test_batch_end], 
            keep_prob:1.0, training:False})
          test_loss += result_loss
          correct += result_correct
        # correct /= 100.0
        test_loss /= 100.0
        if lambda_error == 100.0 :
          lambda_error = ( 100.0 - correct )
        else :
          lambda_error = lambda_error * ( 1.0 - lambda_val ) + ( 100.0 - correct ) * lambda_val
        print ", Test Loss %0.4f"%test_loss,
        print ", Train/Test %5.3f"%(training_loss/test_loss),", Percent Error = %4.2f" % (100.0 - correct),
        print ", Lambda Error %4.2f" % lambda_error,
        print ", ",int(time.time() - start_time),"seconds"
        if training_loss / test_loss < 0.01 :
          print "End condition met.  Rule 'training_loss / test_loss < 0.01'.  Possible overfitting."
          return lambda_error

  return lambda_error


mnist = get_mnist_data()


class simple0 :
  def convolve( self, image, training,  keep_prob ) :
    return conv( image , 1 , 10 , width=28, padding="VALID" )


class simple1 :
  def convolve( self, image, training,  keep_prob ) :
    result = conv( image , 1 , 18 , width=5, stride=2, padding="VALID" )
    return conv( result , 18 , 10 , width=12, padding="VALID" )


class simple2 :
  def convolve( self, image, training,  keep_prob ) :
    result = conv_relu( image , 1 , 18 , width=5, stride=2, padding="VALID" )
    return conv( result , 18 , 10 , width=12, padding="VALID" )


class simple3 :
  def convolve( self, image, training,  keep_prob ) :
    result = conv_relu( image , 1 , 18 , width=5, stride=2, padding="VALID" )
    result = conv( result , 18 , 10 , width=11, padding="VALID" )
    return avg_pool(result)


class simple4 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5, stride=2, padding="VALID" )
    return conv( result , 18 , 10 , width=12, padding="VALID" )


class simple5 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5, padding="VALID" )
    result = max_pool( result ) # 12
    result = conv_relu( result , 18 , 24 , width=5, padding="VALID" )
    result = max_pool( result ) # 4
    result = tf.nn.dropout( result, keep_prob )
    return conv( result , 24 , 10 , width=4, padding="VALID" )

# Epoch   200 , Training Loss 0.036521 , Test Loss 0.0219 , Train/Test 1.665 , Percent Error = 0.70 ,  93 seconds



class simple6 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5, padding="VALID" )
    result = max_pool( result ) # 12
    result = conv_relu( result , 18 , 24 , width=5, padding="VALID" )
    result = max_pool( result ) # 4
    return drop_conv( keep_prob, result , 24 , 10 , width=4, padding="VALID" )

# Epoch   200 , Training Loss 0.032660 , Test Loss 0.0391 , Train/Test 0.835 , Percent Error = 0.66 ,  93 seconds


class simple7 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = conv_relu( result , 18 , 24 , width=5 )
    result = max_pool( result ) # 7
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )

# Epoch   200 , Training Loss 0.018317 , Test Loss 0.0321 , Train/Test 0.571 , Percent Error = 0.67 ,  137 seconds

class simple8 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = drop_conv( keep_prob, result , 18 , 24 , width=5 )
    result = tf.nn.relu( result )
    result = max_pool( result ) # 7
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )
# Epoch 500 Training Loss 0.0460221105441   Percent correct =  0.9927   345 seconds

class simple9 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = resnet_block( result , 18, 3 , training )
    result = conv_relu( result , 18 , 24 , width=5 )
    result = max_pool( result ) # 7
    result = resnet_block( result , 24, 3 , training )
    result = resnet_block( result , 24, 3 , training )
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )
# Epoch 400 Training Loss 7.6104698062e-05  Percent correct =  0.9936   640 seconds

class simple10 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = resnet_block( result , 18, 3 , training )
    result = conv_relu( result , 18 , 24 , width=3 )
    result = max_pool( result ) # 7
    result = resnet_block( result , 24, 3 , training )
    result = resnet_block( result , 24, 3 , training )
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )
# Epoch 800 Training Loss 2.16272041939e-05   Percent correct =  0.9948   1188 seconds


class simple11 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = resnet_block( result , 18, 3 , training )
    result = resnet_block( result , 18, 3 , training )
    result = conv_relu( result , 18 , 24 , width=3 )
    result = max_pool( result ) # 7
    result = resnet_block( result , 24, 3 , training )
    result = resnet_block( result , 24, 3 , training )
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )
# Epoch 800 Training Loss 3.283573418e-06   Percent correct =  0.9943   1565 seconds


class simple12 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5, padding="VALID" )
    result = max_pool( result ) # 12
    result = resnet_block( result , 18, 3 , training )
    result = resnet_block( result , 18, 3 , training )
    result = max_pool( result ) # 6
    result = conv_relu( result , 18 , 24 , width=1 )
    result = resnet_narrow( result , 24, 3 , training )
    result = resnet_narrow( result , 24, 3 , training )
    result = max_pool( result ) # 3
    result = conv_relu( result , 24 , 32 , width=1 )
    result = resnet_narrow( result , 32, 3 , training )
    result = resnet_narrow( result , 32, 3 , training )
    return drop_conv( keep_prob, result , 32 , 10 , width=3, padding="VALID" )
# Epoch 800 Training Loss 0.00631735111412  Percent correct =  0.9868   1315 seconds


class simple13 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=3 )
    result = max_pool( result ) # 14
    result = conv_relu( result , 18 , 24 , width=3 )
    result = max_pool( result ) # 7
    return drop_conv( keep_prob, result , 24 , 10 , width=7, padding="VALID" )
# Epoch 800 Training Loss 0.0190584493312   Percent correct =  0.9926   442 seconds


class simple14 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = tf.nn.relu( drop_conv( keep_prob, result , 18 , 24 , width=5 ) )
    result = max_pool( result ) # 7
    result = tf.nn.relu( drop_conv( keep_prob, result , 24 , 32 , width=5, padding="VALID" ) )
    return conv( result , 32 , 10 , width=3, padding="VALID" )

class simple15 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = conv_relu( result , 18 , 24 , width=5 )
    result = max_pool( result ) # 7
    result = conv_relu( result , 24 , 32 , width=5, padding="VALID" )
    result = tf.nn.dropout( result , keep_prob )
    return conv( result , 32 , 10 , width=3, padding="VALID" )


class simple16 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = max_pool( result ) # 14
    result = conv_relu( result , 18 , 24 , width=5 )
    result = max_pool( result ) # 7
    result = conv_relu( result , 24 , 32 , width=5 )
    result = max_pool( result ) # 4
    result = tf.nn.dropout( result , keep_prob )
    return conv( result , 32 , 10 , width=4, padding="VALID" )



class simple17 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = resnet_narrow( result , 18, 3 , training )
    result = max_pool( result ) # 14
    result = resnet_narrow( result , 18, 3 , training )
    result = conv_relu( result , 18 , 24 , width=5 )
    result = resnet_narrow( result , 24, 3 , training )
    result = max_pool( result ) # 7
    result = resnet_narrow( result , 24, 3 , training )
    result = conv_relu( result , 24 , 32 , width=5, padding="VALID" )
    result = resnet_narrow( result , 32, 3 , training )
    result = tf.nn.dropout( result , keep_prob )
    return conv( result , 32 , 10 , width=3, padding="VALID" )



class simple18 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = resnet_block( result , 18, 3 , training, momentum=0.9 )
    result = max_pool( result ) # 14
    result = resnet_block( result , 18, 3 , training, momentum=0.9 )
    result = conv_relu( result , 18 , 24 , width=5 )
    result = resnet_block( result , 24, 3 , training, momentum=0.9 )
    result = max_pool( result ) # 7
    result = resnet_block( result , 24, 3 , training, momentum=0.9 )
    result = conv_relu( result , 24 , 32 , width=5, padding="VALID" )
    result = resnet_block( result , 32, 3 , training, momentum=0.9 )
    result = tf.nn.dropout( result , keep_prob )
    return conv( result , 32 , 10 , width=3, padding="VALID" )


class simple19 :
  def convolve( self, image, training,  keep_prob ) :
    result = batch_normalization( image , training )
    result = conv_relu( result , 1 , 18 , width=5 )
    result = resnet_block( result , 18, 3 , training, momentum=0.99 )
    result = max_pool( result ) # 14
    result = resnet_block( result , 18, 3 , training, momentum=0.99 )
    result = conv_relu( result , 18 , 24 , width=5 )
    result = resnet_block( result , 24, 3 , training, momentum=0.99 )
    result = max_pool( result ) # 7
    result = resnet_block( result , 24, 3 , training, momentum=0.99 )
    result = conv_relu( result , 24 , 32 , width=5, padding="VALID" )
    result = resnet_block( result , 32, 3 , training, momentum=0.99 )
    result = tf.nn.dropout( result , keep_prob )
    return conv( result , 32 , 10 , width=3, padding="VALID" )



class conv_pool_with_msra :
  def convolve( self, image, training, keep_prob ) :
    layers = [1,32,64]
    width=28
    conv_window=5
    feature_layer_size=128  # maybe 1024
    result = image

    for index in range(len(layers)-1) :
        result = conv_relu( result , layers[index] , layers[index+1] , conv_window )
        result = resnet_block( result , layers=layers[index+1], width=conv_window, training=training )
        result = resnet_block( result , layers=layers[index+1], width=conv_window, training=training )
        result = max_pool( result )
        width   = int(round(width / 2.0))

    result = conv_relu( result , layers[-1] , feature_layer_size , width=width , padding='VALID' )

    h_out        = tf.reshape(result, [-1,feature_layer_size])
    h_out_drop   = tf.nn.dropout(h_out, keep_prob)
    y            = fully_connected(h_out_drop, feature_layer_size, 10)

    return y



def compareModels() :

  model_list = [ simple0(), simple1(), simple2(), simple3(), simple4(), simple5(),
    simple6(), simple7(), simple8(), simple9(), simple10(), simple11(),
    simple12(), simple13(), simple14(), simple15(), simple16(), simple17(),
    simple18(), simple19() ]

  model_list = []

  print "Long run"
  results = []

  for item in model_list :
    results += [ item.__class__.__name__ , train_model_and_report( mnist, item , epochs=200 , learning_rate_value=1e-3) ]


  print """

  ====================================
        FINAL RESULTS
  ====================================

  """

  for item in results :
    print item


  print """
  RESULTS

  0.459933434452      simple18
  0.481827340846      simple9
  0.484590395482      simple11
  0.517686724076      simple19
  0.523299952174      simple10
  0.608307233759      simple16
  0.646958202629      simple15
  0.651675411848      simple8
  0.667441813173      simple7
  0.681077655134      simple14
  0.682826670507      simple5
  0.707398648726      simple6
  0.89458271703       simple13
  0.997465072211      simple12
  1.24220456685       simple3
  1.3567156149        simple2
  1.53155471589       simple4
  7.31041503028       simple1
  7.33090554099       simple0
  88.6500002875       simple17

  """